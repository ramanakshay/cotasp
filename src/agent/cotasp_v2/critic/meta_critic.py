from typing import Optional, Sequence

import distrax
import jax
import flax.linen as nn
import jax.numpy as jnp

from agent.networks.mlp import MLP, _flatten_dict
from agent.networks.constants import default_init, ones_init
from jax import custom_jvp

@custom_jvp
def clip_fn(x):
    return jnp.minimum(jnp.maximum(x, 0), 1.0)

@clip_fn.defjvp
def f_jvp(primals, tangents):
    # Custom derivative rule for clip_fn
    # x' = 1, when 0 < x < 1;
    # x' = 0, otherwise.
    x, = primals
    x_dot, = tangents
    ans = clip_fn(x)
    ans_dot = jnp.where(x >= 1.0, 0, jnp.where(x <= 0, 0, 1.0)) * x_dot
    return ans, ans_dot

def ste_step_fn(x):
    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    # Straight-through estimator of step function
    # its derivative is equal to 1 when 0 < x < 1, 0 otherwise.
    zero = clip_fn(x) - jax.lax.stop_gradient(clip_fn(x))
    return zero + jax.lax.stop_gradient(jnp.heaviside(x, 0))


class MetaCritic(nn.Module):
    hidden_dims: Sequence[int]
    num_tasks: int
    dropout_rate: Optional[float] = None

    def setup(self):
        # TODO: do we really need setup?
        self.backbones = [nn.Dense(h, kernel_init=default_init()) for h in self.hidden_dims]
        self.embeds_bb = [nn.Embed(self.num_tasks, h, embedding_init=default_init()) for h in self.hidden_dims]
        self.values_layer = nn.Dense(1, kernel_init=default_init())

    def __call__(
            self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        x = {"states": observations, "actions": actions}
        for i, layer in enumerate(self.backbones):
            # backbone
            x = layer(x)

            # apply mask
            phi = ste_step_fn(self.embeds_bb[i](task_id))
            mask = jnp.broadcast_to(phi, x.shape)
            x *= mask

            # relu activation
            x = nn.relu(x)

            # dropout
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(
                    x, deterministic=not training
                )

        values = self.values_layer(x)

        return jnp.squeeze(values, -1)



class MetaCriticEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    num_tasks: int
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions, training: bool = False):

        VmapCritic = nn.vmap(
            MetaCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = VmapCritic(self.hidden_dims, self.num_tasks)(
            states, actions, training
        )
        return qs