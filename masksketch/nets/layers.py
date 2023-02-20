# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common layers and blocks."""

import functools
from typing import Any, Tuple

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def _l2_normalize(x, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.

  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def get_norm_layer(train, dtype, norm_type='BN'):
  """Normalization layer."""
  if norm_type == 'BN':
    norm_fn = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name=None,
        axis_index_groups=None,
        dtype=jnp.float32)
  elif norm_type == 'LN':
    norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
  elif norm_type == 'GN':
    norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
  else:
    raise NotImplementedError
  return norm_fn


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  To be specific, Flax includes padding cells when taking the average,
  while TF does not.

  Args:
    x: Input tensor
    window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
      2-dim tuple one gets 2d pooling.
    strides: Must have the same dimension as the window_shape.
    padding: Either 'SAME' or 'VALID' to indicate pooling method.

  Returns:
    pooled: Tensor after applying pooling.
  """
  pool_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                   (1,) + window_shape + (1,),
                                   (1,) + strides + (1,), padding)
  pool_denom = jax.lax.reduce_window(
      jnp.ones_like(x), 0.0, jax.lax.add, (1,) + window_shape + (1,),
      (1,) + strides + (1,), padding)
  return pool_sum / pool_denom


def upsample(x, factor=2):
  n, h, w, c = x.shape
  x = jax.image.resize(x, (n, h * factor, w * factor, c), method='nearest')
  return x


def dsample(x):
  return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding='same')

