from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from agent.networks.mlp import MLP

class StateActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
            self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        inputs = {"states": observations, "actions": actions}
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(
            inputs, training=training
        )
        return jnp.squeeze(critic, -1)


class StateActionEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions, training: bool = False):

        VmapCritic = nn.vmap(
            StateActionValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = VmapCritic(self.hidden_dims, activations=self.activations)(
            states, actions, training
        )
        return qs