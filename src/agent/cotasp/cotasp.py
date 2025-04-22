import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Callable
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from agent.cotasp.train_state import MetaTrainState
from data.types import Params, PRNGKey
import jax
import optax
import distrax
import numpy as np
import jax.numpy as jnp
from agent.base import TaskAgent
from agent.cotasp.actor import MetaPolicy, update_actor
from agent.sac.critic import StateActionEnsemble, update_critic, soft_target_update
from agent.sac.temp import Temperature, update_temperature

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

        # init sample
        observations = observation_space.sample()[np.newaxis]
        actions = action_space.sample()[np.newaxis]
        task_ids = jnp.array([0])

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
        self_actor = actor

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