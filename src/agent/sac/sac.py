import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Callable
from flax.core.frozen_dict import FrozenDict
from data.types import Params, PRNGKey
import jax
import optax
import distrax
import numpy as np
import jax.numpy as jnp
from agent.base import TaskAgent
from agent.sac.train_state import TrainState
from agent.sac.actor import NormalTanhPolicy, update_actor
from agent.sac.critic import StateActionEnsemble, update_critic, soft_target_update
from agent.sac.temp import Temperature, update_temperature

@functools.partial(jax.jit, static_argnames=("backup_entropy", "critic_reduction"))
def _update_jit(
        rng: PRNGKey,
        actor: TrainState,
        critic: TrainState,
        target_critic_params: Params,
        temp: TrainState,
        batch: FrozenDict,
        discount: float,
        tau: float,
        target_entropy: float,
        backup_entropy: bool,
        critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        discount,
        backup_entropy=backup_entropy,
        critic_reduction=critic_reduction,
    )
    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(
        temp, actor_info["entropy"], target_entropy
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )

@functools.partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


class SACAgent(TaskAgent):
    def __init__(self, observation_space, action_space, config):
        self.config = config.agent
        self.action_dim = action_space.shape[-1]

        if self.config.target_entropy is None:
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = self.config.target_entropy

        self.backup_entropy = self.config.backup_entropy
        self.critic_reduction = self.config.critic_reduction

        self.tau = self.config.tau
        self.discount = self.config.discount

        observations = observation_space.sample()
        actions = action_space.sample()

        self.dummy_o = observations
        self.dummy_a = actions

        seed = config.system.seed
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # actor
        actor_configs = self.config.actor
        actor_def = NormalTanhPolicy(actor_configs.hidden_dims, self.action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_configs.learning_rate),
        )

        # critic
        critic_configs = self.config.critic
        critic_def = StateActionEnsemble(critic_configs.hidden_dims, num_qs=2)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_configs.learning_rate),
        )
        target_critic_params = copy.deepcopy(critic_params)

        # Temperature
        temp_configs = self.config.temperature
        temp_def = Temperature(temp_configs.init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_configs.learning_rate),
        )

        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        self._rng = rng

    def update(self, batch: FrozenDict, id: int) -> Dict[str, float]:
        (
        new_rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        info,
            ) = _update_jit(
                self._rng,
                self._actor,
                self._critic,
                self._target_critic_params,
                self._temp,
                batch,
                self.discount,
                self.tau,
                self.target_entropy,
                self.backup_entropy,
                self.critic_reduction,
            )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        self._temp = new_temp

        return info

    def sample_actions(self, batch: np.ndarray, id: int) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng, self._actor.apply_fn, self._actor.params, batch
        )
        self._rng = rng
        return np.asarray(actions)

    def start_task(self, id, hint):
        pass

    def reset_agent(self):
        # re-initialize params of critic ,target_critic and temperature
        self._rng, critic_key, temp_key = jax.random.split(self._rng, 3)

        # critic
        critic_configs = self.config.critic
        critic_def = StateActionEnsemble(critic_configs.hidden_dims, num_qs=2)
        critic_params = critic_def.init(critic_key, self.dummy_o, self.dummy_a)["params"]
        self._critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_configs.learning_rate),
        )
        self._target_critic_params = copy.deepcopy(critic_params)

        #  Temperature
        temp_configs = self.config.temperature
        temp_def = Temperature(temp_configs.init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        self._temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_configs.learning_rate),
        )

        # reset optimizer
        self._actor = self._actor.reset_optimizer()

    def end_task(self, id):
        self.reset_agent()