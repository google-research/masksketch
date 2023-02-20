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

r"""MaskGIT Tokenizer based on VQGAN.

This tokenizer is a reimplementation of VQGAN [https://arxiv.org/abs/2012.09841]
with several modifications. The non-local layers are removed from VQGAN for
faster speed.
"""
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from masksketch.libml import losses
from masksketch.nets import layers
import ml_collections


class ResBlock(nn.Module):
  """Basic Residual Block."""
  filters: int
  norm_fn: Any
  conv_fn: Any
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu
  use_conv_shortcut: bool = False

  @nn.compact
  def __call__(self, x):
    input_dim = x.shape[-1]
    residual = x
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)

    if input_dim != self.filters:
      if self.use_conv_shortcut:
        residual = self.conv_fn(
            self.filters, kernel_size=(3, 3), use_bias=False)(
                x)
      else:
        residual = self.conv_fn(
            self.filters, kernel_size=(1, 1), use_bias=False)(
                x)
    return x + residual


class Encoder(nn.Module):
  """Encoder Blocks."""

  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.embedding_dim = self.config.vqvae.embedding_dim
    self.conv_downsample = self.config.vqvae.conv_downsample
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == "relu":
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == "swish":
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x):
    conv_fn = nn.Conv
    norm_fn = layers.get_norm_layer(
        train=self.train, dtype=self.dtype, norm_type=self.norm_type)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    x = conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i < num_blocks - 1:
        if self.conv_downsample:
          x = conv_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
        else:
          x = layers.dsample(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.embedding_dim, kernel_size=(1, 1))(x)
    return x


class Decoder(nn.Module):
  """Decoder Blocks."""

  config: ml_collections.ConfigDict
  train: bool
  output_dim: int = 3
  dtype: Any = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == "relu":
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == "swish":
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x):
    conv_fn = nn.Conv
    norm_fn = layers.get_norm_layer(
        train=self.train, dtype=self.dtype, norm_type=self.norm_type)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    num_blocks = len(self.channel_multipliers)
    filters = self.filters * self.channel_multipliers[-1]
    x = conv_fn(filters, kernel_size=(3, 3), use_bias=True)(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    for i in reversed(range(num_blocks)):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i > 0:
        x = layers.upsample(x, 2)
        x = conv_fn(filters, kernel_size=(3, 3))(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.output_dim, kernel_size=(3, 3))(x)
    return x


class VectorQuantizer(nn.Module):
  """Basic vector quantizer."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x, **kwargs):
    codebook_size = self.config.vqvae.codebook_size
    codebook = self.param(
        "codebook",
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="uniform"),
        (codebook_size, x.shape[-1]))
    codebook = jnp.asarray(codebook, dtype=self.dtype)
    distances = jnp.reshape(
        losses.squared_euclidean_distance(
            jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    encoding_indices = jnp.argmin(distances, axis=-1)
    encodings = jax.nn.one_hot(
        encoding_indices, codebook_size, dtype=self.dtype)
    quantized = self.quantize(encodings)
    result_dict = dict()
    if self.train:
      e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x)**
                               2) * self.config.vqvae.commitment_cost
      q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x))**2)
      entropy_loss = 0.0
      if self.config.vqvae.entropy_loss_ratio != 0:
        entropy_loss = losses.entropy_loss(
            -distances,
            loss_type=self.config.vqvae.entropy_loss_type,
            temperature=self.config.vqvae.entropy_temperature
        ) * self.config.vqvae.entropy_loss_ratio
      e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
      q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
      entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
      loss = e_latent_loss + q_latent_loss + entropy_loss
      result_dict = dict(
          quantizer_loss=loss,
          e_latent_loss=e_latent_loss,
          q_latent_loss=q_latent_loss,
          entropy_loss=entropy_loss)
      quantized = x + jax.lax.stop_gradient(quantized - x)

    result_dict.update({
        "encodings": encodings,
        "encoding_indices": encoding_indices,
        "raw": x,
    })
    return quantized, result_dict

  def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
    codebook = jnp.asarray(
        self.variables["params"]["codebook"], dtype=self.dtype)
    return jnp.dot(z, codebook)

  def get_codebook(self) -> jnp.ndarray:
    return jnp.asarray(self.variables["params"]["codebook"], dtype=self.dtype)

  def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    codebook = self.variables["params"]["codebook"]
    return jnp.take(codebook, ids, axis=0)


class GumbelVQ(nn.Module):
  """Gumbel VQ."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x, *, tau=1.0):
    codebook_size = self.config.vqvae.codebook_size
    codebook = self.param(
        "codebook",
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="uniform"),
        (codebook_size, x.shape[-1]))
    codebook = jnp.asarray(codebook, dtype=self.dtype)
    distances = jnp.reshape(
        losses.squared_euclidean_distance(
            jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    result_dict = dict()
    encoding_indices = jnp.argmin(distances, axis=-1)
    if self.train:
      noise = jax.random.gumbel(
          self.make_rng("rng"), distances.shape, dtype=self.dtype)
      encodings = jax.nn.softmax((-distances + noise) / tau, axis=-1)
      quantized = self.quantize(encodings)
    else:
      encodings = jax.nn.one_hot(
          encoding_indices, codebook_size, dtype=self.dtype)
      quantized = self.quantize(encodings)
    result_dict.update({
        "quantizer_loss": 0.0,
        "encodings": encodings,
        "encoding_indices": encoding_indices,
    })
    return quantized, result_dict

  def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
    codebook = jnp.asarray(
        self.variables["params"]["codebook"], dtype=self.dtype)
    return jnp.dot(z, codebook)

  def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(self.variables["params"]["codebook"], ids, axis=0)


class VQVAE(nn.Module):
  """VQVAE model."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu

  def setup(self):
    """VQVAE setup."""
    if self.config.vqvae.quantizer == "gumbel":
      self.quantizer = GumbelVQ(
          config=self.config, train=self.train, dtype=self.dtype)
    elif self.config.vqvae.quantizer == "vq":
      self.quantizer = VectorQuantizer(
          config=self.config, train=self.train, dtype=self.dtype)
    else:
      raise NotImplementedError
    output_dim = 3
    self.encoder = Encoder(
        config=self.config, train=self.train, dtype=self.dtype)
    self.decoder = Decoder(
        config=self.config,
        train=self.train,
        output_dim=output_dim,
        dtype=self.dtype)

  def encode(self, input_dict):
    image = input_dict["image"]
    encoded_feature = self.encoder(image)
    if self.config.vqvae.quantizer == "gumbel" and self.train:
      quantized, result_dict = self.quantizer(
          encoded_feature, tau=input_dict["tau"])
    else:
      quantized, result_dict = self.quantizer(encoded_feature)
    return quantized, result_dict

  def decode(self, x: jnp.ndarray) -> jnp.ndarray:
    reconstructed = self.decoder(x)
    return reconstructed

  def get_codebook_funct(self):
    return self.quantizer.get_codebook()

  def decode_from_indices(self, inputs):
    if isinstance(inputs, dict):
      ids = inputs["encoding_indices"]
    else:
      ids = inputs
    features = self.quantizer.decode_ids(ids)
    reconstructed_image = self.decode(features)
    return reconstructed_image

  def encode_to_indices(self, inputs):
    if isinstance(inputs, dict):
      image = inputs["image"]
    else:
      image = inputs
    encoded_feature = self.encoder(image)
    _, result_dict = self.quantizer(encoded_feature)
    ids = result_dict["encoding_indices"]
    return ids

  def __call__(self, input_dict):
    quantized = self.encode(input_dict)
    outputs = self.decoder(quantized)
    return outputs

