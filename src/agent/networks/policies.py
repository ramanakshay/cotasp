import functools
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_jvp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from agent.networks.common import MLP, Params, PRNGKey, \
    default_init, activation_fn, MaskedLayerNorm

# from common import MLP, Params, PRNGKey, default_init, \
#     activation_fn, RMSNorm, create_mask, zero_grads


LOG_STD_MAX = 2
LOG_STD_MIN = -2


class MSEPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        actions = nn.Dense(self.action_dim,
                           kernel_init=default_init())(outputs)
        return nn.tanh(actions)


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """Distribution followed by tanh."""

    def __init__(self, distribution, threshold=.999, validate_args=False):
        """Initialize the distribution.
        Args:
          distribution: The distribution to transform.
          threshold: Clipping value of the action when computing the logprob.
          validate_args: Passed to super class.
        """
        super().__init__(
            distribution=distribution,
            bijector=tfb.Tanh(),
            validate_args=validate_args)
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(1. - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = self.distribution.log_cdf(
            -inverse_threshold) - log_epsilon
        self._log_prob_right = self.distribution.log_survival_function(
            inverse_threshold) - log_epsilon

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold, self._log_prob_left,
            jnp.where(event >= self._threshold, self._log_prob_right,
                      super().log_prob(event)))

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed=None):
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0)

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype,
                                                      num_classes=num_classes)
        del td_properties['bijector']
        return td_properties


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    clip_mean: float = 1.0
    tanh_squash: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.Distribution:
        h = MLP(self.hidden_dims,
                activations=activation_fn(self.name_activation),
                activate_final=True,
                use_layer_norm=self.use_layer_norm)(observations)

        means = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_fc_init_scale))(h)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale))(h)
        else:
            log_stds = self.param(
                'log_stds', nn.initializers.zeros, (self.action_dim,)
            )

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        # Avoid numerical issues by limiting the mean of the Gaussian
        # to be in [-clip_mean, clip_mean]
        # means = jnp.where(
        #     means > self.clip_mean, self.clip_mean,
        #     jnp.where(means < -self.clip_mean, -self.clip_mean, means)
        # )

        # numerically stable method
        base_dist = tfd.Normal(loc=means, scale=jnp.exp(log_stds) * temperature)

        if self.tanh_squash:
            return tfd.Independent(TanhTransformedDistribution(base_dist),
                                   reinterpreted_batch_ndims=1)
        else:
            return base_dist, {'means': means, 'stddev': jnp.exp(log_stds)}


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

def sigma_activation(sigma, sigma_min=LOG_STD_MIN, sigma_max=LOG_STD_MAX):
    return sigma_min + 0.5 * (sigma_max - sigma_min) * (jnp.tanh(sigma) + 1.)

def mu_activation(mu):
    return jnp.tanh(mu)


class MetaPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    task_num: int
    state_dependent_std: bool = True
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = False
    final_fc_init_scale: float = 1.0
    clip_mean: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash: bool = True

    def setup(self):
        self.backbones = [nn.Dense(hidn, kernel_init=default_init()) \
            for hidn in self.hidden_dims]
        self.embeds_bb = [nn.Embed(self.task_num, hidn, embedding_init=default_init()) \
            for hidn in self.hidden_dims]

        self.mean_layer = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_fc_init_scale),
            use_bias=False)

        if self.state_dependent_std:
            self.log_std_layer = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale),
            )
        else:
            self.log_std_layer = self.param(
                'log_std_layer', nn.initializers.zeros,
                (self.action_dim,)
            )

        self.activation = activation_fn(self.name_activation)
        self.tanh = activation_fn('tanh')
        if self.use_layer_norm:
            self.masked_ln = MaskedLayerNorm(use_bias=False, use_scale=False)

    def __call__(self,
                 x: jnp.ndarray,
                 t: jnp.ndarray,
                 temperature: float = 1.0):
        masks = {}
        for i, layer in enumerate(self.backbones):
            x = layer(x)
            # straight-through estimator
            phi_l = ste_step_fn(self.embeds_bb[i](t))
            mask_l = jnp.broadcast_to(phi_l, x.shape)
            masks[layer.name] = mask_l
            # masking outputs
            x *= mask_l
            if self.use_layer_norm and i == 0:
                # layer-normalize output
                x = self.masked_ln(x, mask_l)
                x = self.tanh(x)
            else:
                x = self.activation(x)

        means = self.mean_layer(x)

        # Avoid numerical issues by limiting the mean of the Gaussian
        # to be in [-clip_mean, clip_mean]
        # means = self.hard_tanh(means)
        means = mu_activation(means) * self.clip_mean

        if self.state_dependent_std:
            log_stds = self.log_std_layer(x)
        else:
            log_stds = self.log_std_layer

        # squashing log_std
        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = sigma_activation(log_stds, log_std_min, log_std_max)

        # numerically stable method
        base_dist = tfd.Normal(loc=means, scale=jax.nn.softplus(log_stds) * temperature)

        if self.tanh_squash:
            return tfd.Independent(TanhTransformedDistribution(base_dist),
                                   reinterpreted_batch_ndims=1), {
                                    'masks': masks,
                                    'means': means,
                                    'stddev': jax.nn.softplus(log_stds)
                                   }
        else:
            return base_dist, {'masks': masks, 'means': means, 'stddev': jax.nn.softplus(log_stds)}

    def get_grad_masks(self, masks: dict, input_dim: int = 12):
        grad_masks = {}
        for i, layer in enumerate(self.backbones):
            if i == 0:
                post_m = masks[layer.name]
                grad_masks[(layer.name, 'kernel')] = 1 - jnp.broadcast_to(
                    post_m, (input_dim, self.hidden_dims[i])
                )
                grad_masks[(layer.name, 'bias')] = 1 - post_m.flatten()
                pre_m = masks[layer.name]
            else:
                post_m = masks[layer.name]
                grad_masks[(layer.name, 'kernel')] = 1 - jnp.minimum(
                    jnp.broadcast_to(pre_m.reshape(-1, 1), (self.hidden_dims[i-1], self.hidden_dims[i])),
                    jnp.broadcast_to(post_m, (self.hidden_dims[i-1], self.hidden_dims[i]))
                )
                grad_masks[(layer.name, 'bias')] = 1 - post_m.flatten()
                pre_m = masks[layer.name]

        grad_masks[(self.mean_layer.name, 'kernel')] = 1 - jnp.broadcast_to(
            pre_m.reshape(-1, 1), (self.hidden_dims[-1], self.action_dim)
        )

        return grad_masks


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 5
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        logits = nn.Dense(self.action_dim * self.num_components,
                          kernel_init=default_init())(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         kernel_init=default_init(),
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components,
                            kernel_init=default_init())(outputs)

        shape = list(observations.shape[:-1]) + [-1, self.num_components]
        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        log_stds = jnp.reshape(log_stds, shape)

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        components_distribution = tfd.Normal(loc=mu,
                                             scale=jnp.exp(log_stds) *
                                             temperature)

        base_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=components_distribution)

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())

        return tfd.Independent(dist, 1)


@functools.partial(
    jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == 'det':
        return rng, actor_apply_fn({'params': actor_params}, observations,
                                   temperature)
    else:
        dist = actor_apply_fn(
            {'params': actor_params}, observations, temperature)

    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


def sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_apply_fn, actor_params, observations,
                           temperature, distribution)