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

class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(
            self,
            loc: jnp.ndarray,
            scale_diag: jnp.ndarray,
            low: Optional[jnp.ndarray] = None,
            high: Optional[jnp.ndarray] = None,
    ):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


class MetaPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_tasks: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None

    def setup(self):
        # TODO: do we really need setup?
        self.backbones = [nn.Dense(h, kernel_init=default_init()) for h in self.hidden_dims]
        self.embeds_bb = [nn.Embed(self.num_tasks, h, embedding_init=default_init()) for h in self.hidden_dims]
        self.means_layer = nn.Dense(self.action_dim, kernel_init=default_init())
        self.log_stds_layer = nn.Dense(self.action_dim, kernel_init=default_init())

    def __call__(
            self, observations: jnp.ndarray, task_id: jnp.ndarray, training: bool = False
        ) -> distrax.Distribution:
        x = _flatten_dict(observations)
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

        means = self.means_layer(x)

        log_stds = self.log_stds_layer(x)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return TanhMultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high
        )