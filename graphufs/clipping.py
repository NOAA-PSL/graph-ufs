# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Gradient clipping transformations, copied from optax v0.2.2.

Note that complex numbers are also supported, see
https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
"""

import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import linear_algebra

ClipByGlobalNormState = dict


def clip_by_global_norm(max_norm: float) -> base.GradientTransformation:
  """Clips updates using their global norm.

  Note:
    The difference between this and optax is that this keeps track of g_norm.
    Use at your own risk, version might be out of date and
    I can't get this to work without it having to recompile after first optim iteration.

  References:
    [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

  Args:
    max_norm: The maximum global norm for an update.

  Returns:
    A `GradientTransformation` object.
  """
  global_state = {"g_norm": jnp.asarray(0.)}

  def init_fn(params):
    del params
    return global_state

  def update_fn(updates, state, params=None):
    del params
    g_norm = linear_algebra.global_norm(updates)
    state["g_norm"] = jnp.asarray(g_norm)
    # TODO(b/163995078): revert back to the following (faster) implementation
    # once analysed how it affects backprop through update (e.g. meta-gradients)
    # g_norm = jnp.maximum(max_norm, g_norm)
    # updates = jax.tree_util.tree_map(
    #     lambda t: (t / g_norm) * max_norm, updates)
    trigger = jnp.squeeze(g_norm < max_norm)
    chex.assert_shape(trigger, ())  # A scalar.

    def clip_fn(t):
      return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_norm)

    updates = jax.tree_util.tree_map(clip_fn, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)
