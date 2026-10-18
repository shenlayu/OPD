# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Utilities to create common models from huggingface
"""

import json
import os
import re
from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)







def update_model_config(module_config, override_config_kwargs):
    """Update the module config with the override_config_kwargs.
    Args:
        module_config: The module config from Huggingface Transformers.
        override_config_kwargs: The kwargs to override the module config.
    """
    for key, val in override_config_kwargs.items():
        if isinstance(val, dict):
            update_model_config(getattr(module_config, key), val)
        else:
            setattr(module_config, key, val)


def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> dict:
    if override_config_kwargs is None:
        override_config_kwargs = {}
    assert isinstance(override_config_kwargs, dict), (
        f"override_config_kwargs must be a dict, got {type(override_config_kwargs)}"
    )
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    update_model_config(module_config, override_config_kwargs)

    return module_config


def get_generation_config(
    model: str,
    trust_remote_code: bool = False,
) -> Optional[GenerationConfig]:
    try:
        return GenerationConfig.from_pretrained(model)
    except OSError:  # Not found
        try:
            config = get_huggingface_actor_config(
                model,
                trust_remote_code=trust_remote_code,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None






def get_model_size(model: nn.Module, scale="auto"):
    n_params = sum(p.numel() for p in model.parameters())

    if scale == "auto":
        if n_params > 1e9:
            scale = "B"
        elif n_params > 1e6:
            scale = "M"
        elif n_params > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        n_params = n_params / 1e9
    elif scale == "M":
        n_params = n_params / 1e6
    elif scale == "K":
        n_params = n_params / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}")

    return n_params, scale


def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale="auto")
    if name is None:
        name = model.__class__.__name__
    print(f"{name} contains {n_params:.2f}{scale} parameters")




def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def convert_weight_keys(state_dict: dict[str, torch.Tensor], model: PreTrainedModel):
    # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return state_dict

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    original_weights = {}
    for key, value in state_dict.items():
        for pattern, replacement in reverse_key_mapping.items():
            replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            # Early exit of the loop
            if n_replace > 0:
                break

        original_weights[key] = value

    return original_weights


def check_exclude_modules(config, key: str) -> bool:
    """
    A helper method to check if the passed module's key name matches any of the exclude modules in the adapter_config.
    Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/tuners_utils.py

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match exclude modules from
        key (`str`): A key to search any matches in config

    Returns:
        True of match object if key matches any exclude modules from config, False if no match found
    """
    if hasattr(config, "exclude_modules") and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            if re.fullmatch(config.exclude_modules, key):
                return True
        elif key in config.exclude_modules:
            return True
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):
            return True
    return False


def check_target_modules(config, key: str) -> bool:
    """
    A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.
    Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/tuners_utils.py

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        True of match object if key matches any target modules from config, False if no match found
    """
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    elif key in config.target_modules:
        # this module is specified directly in target_modules
        target_module_found = True
    else:
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave
            # For now, empty layers_pattern means any layer pattern is ok
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes

    return target_module_found


















# pad input_ids_rmpad, cu_seqlens and max_seqlen_in_batch to be divisible by tp










_architecture_to_auto_class = {
    "ForCausalLM": AutoModelForCausalLM,
    "ForVision2Seq": AutoModelForVision2Seq,
    "ForTokenClassification": AutoModelForTokenClassification,
    "ForSequenceClassification": AutoModelForSequenceClassification,
}




def extract_multi_modal_inputs(
    batch_data: list[dict[str, torch.Tensor]],
    indices: Optional[list[int]] = None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """
    Extract and process multi-modal inputs from a batch.

    Args:
        batch_data (list[dict[str, torch.Tensor]]): The batch containing potential multi-modal inputs
        indices (Optional[list[int]]): If provided, only extract inputs at these indices

    Returns:
        dict[str, torch.Tensor | list[torch.Tensor]]: Processed multi-modal inputs ready for model consumption

    """
    multi_modal_inputs = {}
    multi_modal_inputs_collected = {}
    has_image_bound = False

    selected_batch_data = batch_data
    if indices is not None:
        selected_batch_data = [batch_data[i] for i in indices if i < len(batch_data)]

    for inputs in selected_batch_data:
        if "image_bound" in inputs:
            has_image_bound = True
        for key, value in inputs.items():
            if value is not None:
                if key not in multi_modal_inputs_collected:
                    multi_modal_inputs_collected[key] = []
                multi_modal_inputs_collected[key].append(value)

    for key, values in multi_modal_inputs_collected.items():
        if has_image_bound:  # minicpm-o logic
            multi_modal_inputs[key] = values
        else:
            multi_modal_inputs[key] = torch.cat(values, dim=0)

    return multi_modal_inputs


def get_lora_rank_from_adapter(adapter_path: str | os.PathLike) -> int:
    """
    Extract LoRA rank from adapter configuration file.

    Args:
        adapter_path: Path to LoRA adapter directory

    Returns:
        LoRA rank value from adapter_config.json

    Raises:
        FileNotFoundError: If adapter path or config file doesn't exist
        ValueError: If config file is invalid or missing rank
    """
    adapter_path = os.path.abspath(os.path.expanduser(str(adapter_path)))

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            if "r" not in config:
                raise ValueError(f"LoRA rank 'r' not found in {config_path}")
            return int(config["r"])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}") from e
    except (KeyError, ValueError) as e:
        raise ValueError(f"Cannot parse LoRA rank from {config_path}: {e}") from e


