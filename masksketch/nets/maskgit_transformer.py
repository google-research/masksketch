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

r"""MaskGIT Transformer for masked visual token modeling (MVTM) based on BERT.

The transformer is implemented based on a simplified version of BERT
[https://arxiv.org/abs/1810.04805]. Specifically, the part on next sentence
prediction and segment ids are removed from BERT. Taking the masked tokens as
inputs, the model predicts the probability of all individual tokens.

For details, please see https://arxiv.org/abs/2012.09841.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Text, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp


LAYERNORM_EPSILON = 1e-12  # Layer norm from BERT

InitializerType = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]


def truncated_normal(stddev: Union[float, jnp.ndarray], dtype=jnp.float32):

  def init(key: jnp.ndarray, shape: Iterable[int], dtype: jnp.dtype = dtype):
    return jax.random.truncated_normal(
        key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev

  return init


class Attention(nn.Module):
  """Attention layer that is part of each Transformer layer."""
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  hidden_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input: jnp.ndarray, input_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    attention_mask = nn.make_attention_mask(input_mask, input_mask)
    attention_output = nn.attention.SelfAttention(
        num_heads=self.num_attention_heads,
        qkv_features=self.hidden_size,
        dropout_rate=self.attention_probs_dropout_prob,
        deterministic=deterministic,
        kernel_init=self.initializer_fn,
        bias_init=jax.nn.initializers.zeros,
        name='self_attention',
    )(layer_input, attention_mask)

    attention_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        attention_output, deterministic=deterministic)
    attention_output = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='attention_output_ln')(
            attention_output + layer_input)

    return attention_output


class Mlp(nn.Module):
  """MLP layer that is part of each Transformer layer."""
  hidden_size: int
  hidden_dropout_prob: float
  intermediate_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, attention_output: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    intermediate_output = nn.Dense(
        features=self.intermediate_size,
        kernel_init=self.initializer_fn,
        name='intermediate_output')(
            attention_output)
    intermediate_output = jax.nn.gelu(intermediate_output)

    layer_output = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='layer_output')(
            intermediate_output)
    layer_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        layer_output, deterministic=deterministic)
    layer_output = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='layer_output_ln')(
            layer_output + attention_output)

    return layer_output


class TransformerLayer(nn.Module):
  """A single Transformer layer."""
  intermediate_size: int
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input: jnp.ndarray, input_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    attention_output = Attention(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        num_attention_heads=self.num_attention_heads,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_fn=self.initializer_fn)(
            layer_input=layer_input,
            input_mask=input_mask,
            deterministic=deterministic)

    layer_output = Mlp(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        intermediate_size=self.intermediate_size,
        initializer_fn=self.initializer_fn)(
            attention_output=attention_output, deterministic=deterministic)

    return layer_output


class Embed(nn.Module):
  """Embeds visual tokens."""
  embedding_size: int
  hidden_dropout_prob: float
  vocab_size: int
  max_position_embeddings: int
  initializer_fn: InitializerType
  hidden_size: Optional[int] = None

  @nn.compact
  def __call__(self, input_ids: jnp.ndarray,
               deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    seq_length = input_ids.shape[-1]
    position_ids = jnp.arange(seq_length)[None, :]

    word_embedder = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='word_embeddings')
    word_embeddings = word_embedder(input_ids)
    position_embeddings = nn.Embed(
        num_embeddings=self.max_position_embeddings,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='position_embeddings')(
            position_ids)

    input_embeddings = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='embeddings_ln')(
            word_embeddings + position_embeddings)
    if self.hidden_size:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='embedding_hidden_mapping')(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings


class Bias(nn.Module):
  """Adds a learnable bias to the input.

  Attributes:
    dtype: the dtype of the computation (default: float32).
    bias_init: initializer function for the bias.
  """
  dtype: Any = jnp.float32
  bias_init: Callable[[Any, Tuple[int], Any], Any] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)

    bias_shape = inputs.shape[-1]
    bias = self.param('bias', self.bias_init, bias_shape)
    bias = jnp.asarray(bias, self.dtype)
    bias = jnp.broadcast_to(bias, inputs.shape)

    return inputs + bias


class MlmLayer(nn.Module):
  """MLM layer for masked token prediction."""
  hidden_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, last_layer: jnp.ndarray,
               embeddings: jnp.ndarray) -> jnp.ndarray:
    mlm_hidden = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='mlm_dense')(
            last_layer)
    mlm_hidden = jax.nn.gelu(mlm_hidden)
    mlm_hidden = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='mlm_ln')(
            mlm_hidden)
    output_weights = jnp.transpose(embeddings)
    logits = jnp.matmul(mlm_hidden, output_weights)
    logits = Bias(name='mlm_bias')(logits)
    return logits


class Transformer(nn.Module):
  """Transformer modified from BERT."""
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 256
  initializer_range: float = 0.02

  @nn.compact
  def __call__(self,
               input_ids: jnp.ndarray,
               deterministic: bool = True) -> Dict[Text, jnp.ndarray]:
    input_ids = input_ids.astype('int32')
    input_embeddings = Embed(
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings,
        initializer_fn=truncated_normal(self.initializer_range))(
            input_ids=input_ids, deterministic=deterministic)

    layer_input = input_embeddings
    for _ in range(self.num_hidden_layers):
      layer_output = TransformerLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range))(
              layer_input=layer_input,
              input_mask=jnp.ones_like(input_ids, dtype=jnp.int32),
              deterministic=deterministic)
      layer_input = layer_output

    word_embedding_matrix = self.variables['params']['Embed_0'][
        'word_embeddings']['embedding']
    logits = MlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=truncated_normal(self.initializer_range))(
            last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits

