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
from flax.core import FrozenDict
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT


class TrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    Synopsis::

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx)
        grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        for batch in data:
        grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
    step: Counter starts at 0 and is incremented by every call to
        `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
        convenience to have a shorter params list for the `train_step()` function
        in your training loop.
    params: The parameters to be updated by `tx` and used by `apply_fn`.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
    """
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

Note that internally this function calls `.tx.update()` followed by a call
to `optax.apply_updates()` to update `params` and `opt_state`.

Args:
grads: Gradients that have the same pytree structure as `.params`.
**kwargs: Additional dataclass attributes that should be `.replace()`-ed.

Returns:
An updated instance of `self` with `step` incremented by one, `params`
and `opt_state` updated by applying `grads`, and additional attributes
replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def update_params(self, new_params: core.FrozenDict[str, Any]) -> 'TrainState':
        return self.replace(params=new_params)

    def reset_optimizer(self) -> 'TrainState':
        # contain the count argument
        init_opt_state = tree_map(
            lambda x: jnp.zeros_like(x), self.opt_state
        )
        return self.replace(opt_state=init_opt_state)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'TrainState':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
            params = tree_map(jnp.array, params)
        return self.replace(params=params)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs
        )
        