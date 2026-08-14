# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from copy import deepcopy

from gr00t.configs.smpl_root_mode import (
    assert_resume_root_setup_matches,
    patch_modality_configs_for_root_mode,
)
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    setup = ft_config.smpl_root_setup
    assert setup is not None
    print(
        f"SMPL root training: mode={setup.root_process_mode!r}, "
        f"action_mode={setup.action_mode!r}, "
        f"use_relative_euler={setup.use_relative_euler}, "
        f"use_state_euler={setup.use_state_euler}, "
        f"state.robot_root={setup.include_state_robot_root}"
    )

    if ft_config.resume_from_checkpoint:
        assert_resume_root_setup_matches(
            output_dir=ft_config.output_dir,
            experiment_name=ft_config.experiment_name,
            setup=setup,
            embodiment_tag=embodiment_tag,
        )

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.data.modality_configs = patch_modality_configs_for_root_mode(
        deepcopy(config.data.modality_configs),
        embodiment_tag,
        setup,
    )

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    # Disable all image augmentations for fine-tuning: no random crop / rotation /
    # color jitter / mask-based extras. Keep only deterministic letterbox + resize.
    config.model.random_rotation_angle = None
    config.model.color_jitter_params = None
    config.model.extra_augmentation_config = None
    config.model.shortest_image_edge = None
    config.model.crop_fraction = None
    if config.model.image_target_size is not None:
        config.model.image_crop_size = tuple(config.model.image_target_size)

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.use_relative_euler = ft_config.use_relative_euler
    config.model.use_state_euler = ft_config.use_state_euler

    # Prefer HF cache under CACHE_ROOT (set by train.sh); avoid silent /root/.cache hits.
    # model_name must be a local dir when offline — HF repo IDs still probe the Hub
    # (e.g. adapter_config.json HEAD) even if weights are already cached.
    hf_hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        config.training.transformers_cache_dir = hf_hub_cache

    # Training Cosmos path: train.sh pins /sh/ycb HF cache via COSMOS_REASON2_PATH
    # / HF_HUB_CACHE. Do not use the laptop ~/.cache path here.
    from gr00t import resolve_cosmos_reason2_path

    cosmos_repo = "nvidia/Cosmos-Reason2-2B"
    cosmos_local = resolve_cosmos_reason2_path()
    if cosmos_local:
        config.model.model_name = cosmos_local
        config.training.transformers_local_files_only = True
        print(f"Using training Cosmos backbone: {cosmos_local}")
    else:
        config.model.model_name = cosmos_repo
        print(
            f"WARNING: Cosmos cache not found under HF_HUB_CACHE/COSMOS_REASON2_PATH; "
            f"will use Hub id {cosmos_repo}. On the training server run via train.sh "
            "so HF_HUB_CACHE=/sh/ycb/.cache/huggingface/hub."
        )

    # Align model max action horizon with the registered modality action chunk length.
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    action_modality = MODALITY_CONFIGS[embodiment_tag]["action"]
    config.model.action_horizon = len(action_modality.delta_indices)
    print(
        f"Set model.action_horizon={config.model.action_horizon} "
        f"from modality action delta_indices"
    )

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.use_tensorboard = ft_config.use_tensorboard
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    # Dataset videos are AV1; torchcodec/system ffmpeg lack a software decoder here.
    config.data.video_backend = "pyav"

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
