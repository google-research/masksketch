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

r"""Configuration and hyperparameter sweeps for maskgit training."""

from masksketch.configs import base_config
from masksketch.configs import vqgan_config
from masksketch.configs import maskgit_class_cond_config
import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = maskgit_class_cond_config.get_config()
  config.experiment = "masksketch_class_cond"
  config.model_class = "maskgit_class_cond"
  config.sequence_order = "horizontal"

  config.num_class = 1000
  config.eval_batch_size = 8
  config.image_size = 256

  config.label_smoothing = 0.
  config.mask_scheduling_method = "uniform"
  config.sample_num_iterations = 1000
  config.sample_choice_temperature = 0.
  config.label_smoothing = 0.
  config.guidance_scale = 0.6


  config.structure = ml_collections.ConfigDict()
  config.structure.layers = (1, 3, 16, 20, 21, 22)
  config.structure.min_mask_rate = 0.35
  config.structure.max_mask_rate = 0.95
  config.structure.lambda_structure = 0.9
  config.structure.temperature_structure = 0.
  config.structure.temperature_confidence = 0.

  return config


def get_hyper(h):
  return h.product([
      h.sweep("image_size", [256, 512]),
      h.sweep("compute_loss_for_all", [True, False]),
  ],
                   name="config")
