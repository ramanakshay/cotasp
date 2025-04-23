import copy
import itertools
import functools
from typing import Dict, Optional, Sequence, Tuple, Callable
from flax.core.frozen_dict import FrozenDict
from agent.cotasp.train_state import TrainState, MetaTrainState
from data.types import Params, PRNGKey
import jax
import optax
import distrax
import numpy as np
import jax.numpy as jnp
from agent.base import TaskAgent
from agent.cotasp.actor import MetaPolicy, update_theta, update_alpha
from agent.cotasp.critic import StateActionEnsemble, update_critic, soft_target_update
from agent.cotasp.temp import Temperature, update_temperature
from agent.cotasp.dict_learner import OnlineDictLearnerV2
from sentence_transformers import SentenceTransformer

@functools.partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray,
        task_id: jnp.ndarray
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations, task_id)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


@functools.partial(jax.jit, static_argnames=("backup_entropy", "critic_reduction"))
def _update_cotasp_jit(
        rng: PRNGKey,
        actor: TrainState,
        critic: TrainState,
        target_critic_params: Params,
        temp: TrainState,
        batch: FrozenDict,
        task_id: jnp.ndarray,
        optimize_alpha: bool,
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
        task_id,
        discount,
        backup_entropy,
        critic_reduction,
    )

    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    rng, key = jax.random.split(rng)
    new_actor, actor_info = jax.lax.cond(
        optimize_alpha,
        update_alpha,
        update_theta,
        key, actor, new_critic, temp, batch, task_id
    )
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

class CoTASPAgent(TaskAgent):
    def __init__(self, observation_space, action_space, num_tasks, config):
        self.config = config.agent
        self.num_tasks = num_tasks
        self.action_dim = action_space.shape[-1]

        if self.config.target_entropy is None:
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = self.config.target_entropy

        self.backup_entropy = self.config.backup_entropy
        self.critic_reduction = self.config.critic_reduction

        self.tau = self.config.tau
        self.discount = self.config.discount

        self.schedule = itertools.cycle([False]*self.config.theta_steps + [True]*self.config.alpha_steps)
        self.update_dict = self.config.update_dict

        # init sample
        observations = observation_space.sample()[np.newaxis]
        actions = action_space.sample()[np.newaxis]
        task_ids = jnp.array([0])

        self.dummy_o = observations
        self.dummy_a = actions

        seed = config.system.seed
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # actor
        actor_configs = self.config.actor
        actor_def = MetaPolicy(actor_configs.hidden_dims, self.action_dim, self.num_tasks)
        actor_params = actor_def.init(actor_key, observations, task_ids)["params"]
        actor = MetaTrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_configs.learning_rate),
        )

        # task encoder
        self.task_embeddings = []
        self.task_encoder = SentenceTransformer('all-MiniLM-L12-v2')
        embedding_dim = self.task_encoder.get_sentence_embedding_dimension()

        # dictionary
        self.dict4layers = {}
        dict_configs = dict(self.config.dictionary)
        for i, h in enumerate(actor_configs.hidden_dims):
            dict_learner = OnlineDictLearnerV2(
                embedding_dim,
                h,
                seed+i+1,
                None, # whether using svd dictionary initialization
                **dict_configs)
            self.dict4layers[f'embeds_bb_{i}'] = dict_learner

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

    def sample_actions(self, batch: np.ndarray, id: int) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng, self._actor.apply_fn, self._actor.params, batch, jnp.array([id])
        )
        self._rng = rng
        return np.asarray(actions)

    def update(self, batch: FrozenDict, id: int) -> Dict[str, float]:
        optimize_alpha = next(self.schedule) if self.update_dict else False

        (
        new_rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        info,
            ) = _update_cotasp_jit(
                self._rng,
                self._actor,
                self._critic,
                self._target_critic_params,
                self._temp,
                batch,
                jnp.array([id]),
                optimize_alpha,
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

    def start_task(self, id: int, hint: str):
        task_e = self.task_encoder.encode(hint)[np.newaxis, :]
        self.task_embeddings.append(task_e)

        actor_params = self._actor.params
        for k in self._actor.params.keys():
            if k.startswith('embeds'):
                alpha_l = self.dict4layers[k].get_alpha(task_e)
                alpha_l = jnp.asarray(alpha_l.flatten())
                # Replace the i-th row
                actor_params[k]['embedding'] = actor_params[k]['embedding'].at[id].set(alpha_l)
        self._actor = self._actor.update_params(actor_params)

    def reset_agent(self):
        # re-initialize params of critic ,target_critic and temperature
        self._rng, critic_key, temp_key = jax.random.split(self._rng, 3)

        # critic
        critic_configs = self.config.critic
        critic_def = StateActionEnsemble(critic_configs.hidden_dims, num_qs=2)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
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

    def end_task(self, id: int):
        if self.update_dict:
            for k in self._actor.params.keys():
                if k.startswith('embeds'):
                    optimal_alpha_l = self._actor.params[k]['embedding'][task_id]
                    optimal_alpha_l = np.array([optimal_alpha_l.flatten()])
                    task_e = self.task_embeddings[task_id]
                    # online update dictionary via CD
                    self.dict4layers[k].update_dict(optimal_alpha_l, task_e)
                    dict_stats[k] = {
                        'sim_mat': self.dict4layers[k]._compute_overlapping(),
                        'change_of_d': np.array(self.dict4layers[k].change_of_dict)
                    }
        else:
            for k in self._actor.params.keys():
                if k.startswith('embeds'):
                    dict_stats[k] = {
                        'sim_mat': self.dict4layers[k]._compute_overlapping(),
                        'change_of_d': 0
                    }

        self.reset_agent()

        return dict_stats
