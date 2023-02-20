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

import os
import io
import flax
import functools
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import ImageFilter, Image
import requests
import tensorflow.compat.v1 as tf

from masksketch.nets import vqgan_tokenizer, maskgit_transformer
from masksketch.configs import masksketch_class_cond_config
from masksketch.libml import parallel_decode_with_structure_similarity
from masksketch.utils import restore_from_path
from masksketch.inference import ImageNet_class_conditional_generator


import flax.linen as nn
from flax.linen.dtypes import promote_dtype

def get_kqv(intermediates, num_layers):
  """Given BERT intermediates, returns keys, queries abd values at each layer."""
  out_dict = {x: list(range(num_layers)) for x in ["key", "value", "query"]}
  for layer in intermediates:
    if "TransformerLayer" in layer:
      attn_layer = intermediates[layer]["Attention_0"]["self_attention"]
      layer_num = int(layer.split("_")[1])
      for ft in ["key", "value", "query"]:
        features = attn_layer[ft]["__call__"][0]
        out_dict[ft][layer_num] = features
  for ft in ["key", "value", "query"]:
    out_dict[ft] = jnp.array(out_dict[ft])
  return out_dict

@jax.jit
def get_attn(key, query, attn_mask):
  """Returns self-attention weights givern key, query and the attention mask."""
  mask  = nn.make_attention_mask(attn_mask, attn_mask)
  query, key = promote_dtype(query, key, dtype=jnp.float32)
  attn_weights = nn.dot_product_attention_weights(
      query, key, None, mask, True, None, 0, True,
      jnp.float32, None)
  return attn_weights


@jax.jit
def get_attn_vmap(key_list, query_list, attn_masks):
  layer_attn = jax.vmap(get_attn, 0)(
      key_list, query_list, attn_masks)  # (L x B x h x N x N)
  return jnp.transpose(layer_attn, (1, 0, 2, 3, 4))  # (B x L x h x N x N)


def mask_tokens(rng, tokens, mask_rate=0., mask_token=-1):
  mask_len = jax.numpy.clip(
      jnp.ceil(tokens.shape[-1] * mask_rate),
      a_min=0, a_max=(tokens.shape[-1] - 1))
  fullmask = jnp.full_like(tokens, mask_token)
  rng, subrng = jax.random.split(rng)
  should_mask = jax.random.uniform(subrng, shape=tokens.shape)
  mask = parallel_decode_with_structure_similarity.simple_topk_with_temp(
    subrng, should_mask, mask_len, 0.)
  masked_inputs = jnp.where(mask, fullmask, tokens)
  return masked_inputs


class MaskSketch_generator(ImageNet_class_conditional_generator):

  def __init__(self, image_size=256, config=None):
    super(MaskSketch_generator, self).__init__(image_size=image_size)
    if config is None:
      config = masksketch_class_cond_config.get_config()
    self.maskgit_cf = config
    self.maskgit_cf.image_size = int(image_size)
    self.maskgit_cf.eval_batch_size = 8

  def image_to_tokens(self, rng, image, label, mask_rate=0.):
    imgs = self._create_input_batch(image)

    # Encode the images into image tokens
    image_tokens = self.tokenizer_model.apply(
          self.tokenizer_variables,
          {"image": imgs},
          method=self.tokenizer_model.encode_to_indices,
          mutable=False)
    image_tokens = np.reshape(image_tokens, [self.maskgit_cf.eval_batch_size, -1])
    masked_tokens = mask_tokens(
        rng, image_tokens, mask_rate, self.maskgit_cf.transformer.mask_token_id)

    # Create input tokens based on the category label
    label_tokens = label * jnp.ones([self.maskgit_cf.eval_batch_size, 1])
    # Shift the label tokens by codebook_size
    label_tokens = label_tokens + self.maskgit_cf.vqvae.codebook_size
    # Concatenate the two as input_tokens
    input_tokens = jnp.concatenate([label_tokens, masked_tokens], axis=-1)
    return input_tokens.astype(jnp.int32)

  def generate_samples(self, rng, input_image, class_label, num_iterations=16):

    str_layers = np.array(self.maskgit_cf.structure.layers)
    if str_layers is None:
      str_layers = np.arange(self.maskgit_cf.transformer.num_layers)

    uncond_pad = jax.random.randint(
        rng, (2, 1),
        minval=self.maskgit_cf.vqvae.codebook_size,
        maxval=self.maskgit_cf.vqvae.codebook_size+self.maskgit_cf.num_class
    )

    @jax.jit
    def tokens_to_logits_classifier_free(seq):
      uncond_seq = seq
      uncond_seq = jax.lax.dynamic_update_slice(uncond_seq, uncond_pad, (0, 0))
      all_seq = jnp.concatenate([seq, uncond_seq], axis=0)
      logits = self.transformer_model.apply(
          self.transformer_variables, all_seq, deterministic=True)
      logits = logits[..., :self.maskgit_cf.vqvae.codebook_size]
      cond_logits, uncond_logits = jnp.split(logits, 2, axis=0)
      # Use the original paper (CF-GUIDANCE) setting.
      cond_logits = cond_logits + self.maskgit_cf.guidance_scale * (
          cond_logits - uncond_logits)
      return cond_logits

    @jax.jit
    def tokens_to_attn(seq):
      """Returns logits and attention weights."""
      _, features_state = self.transformer_model.apply(
          self.transformer_variables, seq, deterministic=True,
          capture_intermediates=True, mutable=["intermediates"])
      intermediates = features_state["intermediates"]
      kqv_dict = get_kqv(
          intermediates, self.maskgit_cf.transformer.num_layers)
      attn_masks = jnp.repeat(
          jnp.ones_like(seq)[None,...], len(str_layers), axis=0)
      attn_weights = get_attn_vmap(
          kqv_dict["key"][str_layers],
          kqv_dict["query"][str_layers],
          attn_masks)
      return attn_weights

    rng, mask_rng = jax.random.split(rng)
    input_tokens = self.image_to_tokens(mask_rng, input_image, class_label, 0.)
    masked_tokens = self.image_to_tokens(
        mask_rng, input_image, class_label,
        self.maskgit_cf.structure.max_mask_rate)

    output_tokens = parallel_decode_with_structure_similarity.decode(
          input_tokens,
          masked_tokens,
          rng,
          tokens_to_logits_classifier_free,
          tokens_to_attn,
          num_iter=num_iterations,
          mask_scheduling_method=self.maskgit_cf.mask_scheduling_method,
          structure_params=self.maskgit_cf.structure)

    output_tokens = jnp.reshape(
        # output_tokens[:, -1, 1:],
        output_tokens[:, 1:],
        [-1, self.transformer_latent_size, self.transformer_latent_size])
    gen_images = self.tokenizer_model.apply(
        self.tokenizer_variables,
        output_tokens,
        method=self.tokenizer_model.decode_from_indices,
        mutable=False)

    return gen_images