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

"""Discriminator from StyleGAN."""

import functools
import math
from typing import Any, Tuple

import flax.linen as nn
from flax.linen.initializers import xavier_uniform
import jax
from jax import lax
import jax.numpy as jnp

from masksketch.nets import layers
import ml_collections

default_kernel_init = xavier_uniform()


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class BlurPool2D(nn.Module):
  """A layer to do channel-wise blurring + subsampling on 2D inputs.

  Reference:
    Zhang et al. Making Convolutional Networks Shift-Invariant Again.
    https://arxiv.org/pdf/1904.11486.pdf.
  """
  filter_size: int = 4
  strides: Tuple[int, int] = (2, 2)
  padding: str = 'SAME'

  def setup(self):
    if self.filter_size == 3:
      self.filter = [1., 2., 1.]
    elif self.filter_size == 4:
      self.filter = [1., 3., 3., 1.]
    elif self.filter_size == 5:
      self.filter = [1., 4., 6., 4., 1.]
    elif self.filter_size == 6:
      self.filter = [1., 5., 10., 10., 5., 1.]
    elif self.filter_size == 7:
      self.filter = [1., 6., 15., 20., 15., 6., 1.]
    else:
      raise ValueError('Only filter_size of 3, 4, 5, 6 or 7 is supported.')

    self.filter = jnp.array(self.filter, dtype=jnp.float32)
    self.filter = self.filter[:, None] * self.filter[None, :]
    with jax.default_matmul_precision('float32'):
      self.filter /= jnp.sum(self.filter)
    self.filter = jnp.reshape(
        self.filter, [self.filter.shape[0], self.filter.shape[1], 1, 1])

  @nn.compact
  def __call__(self, inputs):
    channel_num = inputs.shape[-1]
    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    depthwise_filter = jnp.tile(self.filter, [1, 1, 1, channel_num])
    with jax.default_matmul_precision('float32'):
      outputs = lax.conv_general_dilated(
          inputs,
          depthwise_filter,
          self.strides,
          self.padding,
          feature_group_count=channel_num,
          dimension_numbers=dimension_numbers)
    return outputs


class ResBlock(nn.Module):
  """StyleGAN ResBlock for D.

  https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L618
  """
  filters: int
  activation_fn: Any
  blur_resample: bool

  @nn.compact
  def __call__(self, x):
    input_dim = x.shape[-1]
    residual = x
    x = nn.Conv(input_dim, (3, 3), kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    if self.blur_resample:
      x = BlurPool2D(filter_size=4)(x)
      residual = BlurPool2D(filter_size=4)(residual)
    else:
      x = layers.dsample(x)
      residual = layers.dsample(residual)
    residual = nn.Conv(
        self.filters, (1, 1), use_bias=False, kernel_init=default_kernel_init)(
            residual)
    x = nn.Conv(self.filters, (3, 3), kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    out = (residual + x) / math.sqrt(2)
    return out


class Discriminator(nn.Module):
  """StyleGAN Discriminator."""

  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  def setup(self):
    self.input_size = self.config.image_size
    self.blur_resample = self.config.discriminator.blur_resample
    self.activation_fn = functools.partial(
        jax.nn.leaky_relu, negative_slope=0.2)
    self.channel_multiplier = self.config.discriminator.channel_multiplier

  @nn.compact
  def __call__(self, x):
    filters = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256 * self.channel_multiplier,
        128: 128 * self.channel_multiplier,
        256: 64 * self.channel_multiplier,
        512: 32 * self.channel_multiplier,
        1024: 16 * self.channel_multiplier,
    }
    x = nn.Conv(
        filters[self.input_size], (3, 3), kernel_init=default_kernel_init)(
            x)
    x = self.activation_fn(x)
    log_size = int(math.log2(self.input_size))
    for i in range(log_size, 2, -1):
      x = ResBlock(filters[2**(i - 1)], self.activation_fn, self.blur_resample)(
          x)
    x = nn.Conv(filters[4], (3, 3), kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(filters[4], kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    x = nn.Dense(1, kernel_init=default_kernel_init)(x)
    return x


  def squared_euclidean_distance
