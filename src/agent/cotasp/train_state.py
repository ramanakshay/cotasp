# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from collections.abc import Callable

import optax

import jax
from flax import core, struct, traverse_util
from flax.core import freeze, unfreeze, FrozenDict
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT


def filter_theta(path, _):
    for i in range(10):
        if f'backbones_{i}' in path:
            return 'frozen'
    if 'means_layer' in path:
        return 'frozen'
    elif 'log_stds_layer' in path:
        return 'frozen'
    else:
        return 'trainable'


def filter_alpha(path, _):
    for i in range(10):
        if f'embeds_bb_{i}' in path:
            return 'frozen'
    return 'trainable'


class MetaTrainState(struct.PyTreeNode):
    # A simple train state allowing alternate optimization.

    step: int | jax.Array
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx_theta: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_alpha: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state_theta: optax.OptState
    opt_state_alpha: optax.OptState

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_grads_theta(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx_theta.update(
            grads, self.opt_state_theta, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state_theta=new_opt_state,
            **kwargs,
        )

    def apply_grads_alpha(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx_alpha.update(
            grads, self.opt_state_alpha, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state_alpha=new_opt_state,
            **kwargs,
        )

    def update_params(self, new_params: core.FrozenDict[str, Any]) -> 'MPNTrainState':
        return self.replace(params=new_params)

    def reset_optimizer(self):
        # contain the count argument
        opt_state_theta = tree_map(
            lambda x: jnp.zeros_like(x), self.opt_state_theta
        )
        opt_state_alpha = tree_map(
            lambda x: jnp.zeros_like(x), self.opt_state_alpha
        )
        return self.replace(opt_state_theta=opt_state_theta,
                            opt_state_alpha=opt_state_alpha)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'MetaTrainState':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
            params = tree_map(jnp.array, params)
        return self.replace(params=params)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        # opt_state = tx.init(params)
        partition_optimizers = {'trainable': tx, 'frozen': optax.set_to_zero()}

        # theta optimizer
        param_theta = traverse_util.path_aware_map(filter_alpha, params)
        tx_theta = optax.multi_transform(partition_optimizers, param_theta)
        # alpha optimizer
        param_alpha = traverse_util.path_aware_map(filter_theta, params)
        tx_alpha = optax.multi_transform(partition_optimizers, param_alpha)
        
        # init optimizer
        opt_state_theta = tx_theta.init(params)
        opt_state_alpha = tx_alpha.init(params)

        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx_theta=tx_theta,
            tx_alpha=tx_alpha,
            opt_state_theta=opt_state_theta,
            opt_state_alpha=opt_state_alpha,
            **kwargs,
        )