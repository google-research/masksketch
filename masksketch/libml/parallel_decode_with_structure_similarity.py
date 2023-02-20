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

"""Fast decoding routines for non-autoregressive generation."""

import flax
import jax
from jax import lax
import jax.numpy as jnp
from masksketch.libml import mask_schedule
from typing import Iterable
import dataclasses


# Confidence score for known tokens to avoid masking or repredicting them.
# Here we don't use 1.0 because the upper bounder of the probability can be
# possiblity larger than 1 due to the noise addition.
_CONFIDENCE_OF_KNOWN_TOKENS = jnp.inf


@flax.struct.dataclass
class State:
  """Holds decoding state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jnp.DeviceArray  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  cur_seqs: jnp.DeviceArray  # int32 [batch, seq_len]
  rng: jnp.DeviceArray  # Sampling random state.
  final_seqs: jnp.DeviceArray  # int32 [batch, seq_len]


@dataclasses.dataclass
class StructureParams:
  lambda_structure: float = 1.0
  min_mask_rate: float = 0.0
  max_mask_rate: float = 1.0
  layers: Iterable[int] = (0,)
  temperature_structure = 0.
  temperature_confidence = 0.


def state_init(init_indices, rng):
  """Initializes the decoding state data structure."""
  cur_index0 = jnp.array(0)
  return State(
      cur_index=cur_index0, cur_seqs=init_indices, rng=rng, final_seqs=init_indices)


def simple_topk_with_temp(rng, confidence, k, temperature=0.):
  """Simple top-k of confidence values with randomness.

  Arguments:
    rng: a PRNG key used as the random key.
    confidence: confidence scores of each token.
    k: number of unmask tokens.
    temperature: randomness temperature.
  Returns:
    Resulting mask with k unmasked tokens.
  """
  rand_confidence = confidence + temperature * jax.random.gumbel(
      rng, confidence.shape)
  sorted_confidence = jnp.sort(rand_confidence, axis=-1)
  # Obtains cut off threshold given the mask lengths.
  cut_off = jnp.take_along_axis(
      sorted_confidence, (jnp.ones(confidence.shape) * k).astype(jnp.int32),
      axis=-1)
  # Masks tokens with lower confidence.
  masking = (confidence < cut_off)
  return masking


@jax.jit
def self_attention_confidence(target_attn, pred_attn):
  """Returns the self-attention symmetric KL divergence matrix.

  Arguments:
    target_attn: target attention maps of shape (B, L, H, N, N), where B is the
      batch size, L is the number of layers, H is the number of heads, N is the
      token sequence length.
    pred_attn: attention maps of the sampled token sequence. Must have the same
      shape (B, L, H, N, N) as the target_attn.

  Returns:
    Token attention similarity score of shape (B, L, N).
  """
  kl = jnp.sum(target_attn * jnp.log(pred_attn), axis=-1)
  kl_rev = jnp.sum(pred_attn * jnp.log(target_attn), axis=-1)
  return jnp.mean(kl + kl_rev, axis=1)


@jax.jit
def self_attention_confidence_vmapped(target_attn, pred_attn):
  """Returns the self-attention scores over multiple layers."""
  return jax.vmap(self_attention_confidence, 0)(target_attn, pred_attn)


def decode(inputs,
           masked_inputs,
           rng,
           tokens_to_logits,
           tokens_to_attn,
           num_iter=12,
           mask_scheduling_method="cosine",
           structure_params=StructureParams()):
  """Fast decoding for bert iterative generation.

  In the decoding alogrithm, we take iterations to refine them.

  Args:
    inputs: array: [batch_size, length] int32 sequence of unmasked tokens.
    masked_inputs: [batch_size, length] int32 sequence of masked tokens, masked
      tokens have value -1.
    rng: jnp.DeviceArray: sampling random state.
    tokens_to_logits: decoder function taking single token slices and
      cache and returning logits and updated cache.
    tokens_to_attn: decoder function taking single token slices and
      cache and self-attention weights.
    num_iter: default is 12.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.
    structure_params: structure sampling parameters.

  Returns:
     Tuple of:
       [batch_size, max_decode_len] layout sequences
  """
  inputs = inputs.astype("int32")
  masked_inputs = masked_inputs.astype("int32")
  unknown_number_in_the_beginning = jnp.sum(masked_inputs == -1, axis=-1)
  # Initializes state
  init_state = state_init(masked_inputs, rng)
  target_attn = tokens_to_attn(inputs)
  min_mask = structure_params.min_mask_rate
  max_mask = structure_params.max_mask_rate
  temperature_structure = structure_params.temperature_structure
  temperature_confidence = structure_params.temperature_confidence

  def rescale(scores):
    """"Rescaling for the scores."""
    norm_scores = scores - jnp.min(scores, axis=-1, keepdims=True)
    return norm_scores / jnp.max(norm_scores, axis=-1, keepdims=True)

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  def loop_body_fn(state):
    """Beam search loop state update function."""
    rng = state.rng
    step = state.cur_index
    # Current input ids: [batch, seq_len].
    cur_ids = state.cur_seqs
    lambda_s = structure_params.lambda_structure

    # Calls model on current seqs to get next-iteration seqs.
    logits = tokens_to_logits(cur_ids)
    rng, sample_rng = jax.random.split(rng, 2)
    # Samples the ids using categorical sampling: [batch, seq_len].
    sampled_ids = jax.random.categorical(sample_rng, logits)

    # Just updates the masked tokens.
    unknown_map = (cur_ids == -1)
    sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
    attn = tokens_to_attn(sampled_ids)
    struct_scores = self_attention_confidence_vmapped(target_attn, attn)
    # average over layers
    struct_scores = jnp.mean(struct_scores, axis=1)  # (B, L, N) -> (B, N)
    struct_scores = rescale(struct_scores)  # rescaling to [0., 1.]
    # Defines the mask ratio for the next round. The number to mask out is
    # determined by mask_ratio * unknown_number_in_the_beginning.
    ratio = 1. * (step + 1) / num_iter
    mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                        mask_scheduling_method)
    final_seqs = sampled_ids
    mask_ratio = min_mask +  mask_ratio * (max_mask - min_mask)
    # Updates final seqs with the current sampled_ids.
    # Computes the probabilities of each selected tokens.
    probs = jax.nn.softmax(logits, axis=-1)
    selected_probs = jnp.squeeze(
        jnp.take_along_axis(probs,
                            jnp.expand_dims(sampled_ids.astype(jnp.int32), -1),
                            -1), -1)
    selected_probs = selected_probs.at[:, 1:].set(
        rescale(selected_probs[:, 1:]))  # rescaling to [0., 1.]

    # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = jnp.expand_dims(
        jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
    # Adds noise for randomnesss
    # masking = parallel_decode.simple_topk(token_scores, mask_len)
    # Computes the overall token score as a mixture of confidence and
    # structure scores. Note that selected probs do not necessarily keep
    # tokens unmasked at the previous step.
    conf_mask = simple_topk_with_temp(
        sample_rng, selected_probs, mask_len * (1. - lambda_s),
        temperature_confidence)
    structure_mask = simple_topk_with_temp(
        sample_rng, struct_scores, mask_len * lambda_s, temperature_structure)
    masking = jnp.logical_or(structure_mask, conf_mask)

    masking = masking.at[:, 0].set(0)
    # Masks tokens with lower confidence.
    sampled_ids = jnp.where(masking, -1, sampled_ids)
    return State(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.final_seqs