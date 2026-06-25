#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# mkdir -p /data1/ycb/cache/huggingface/hub
# mkdir -p /data1/ycb/cache/huggingface/datasets
mkdir -p /data1/ycb/cache/xdg

export HF_HOME=/data1/ycb/cache/huggingface
export HF_HUB_CACHE=/data1/ycb/cache/huggingface/hub
export HF_DATASETS_CACHE=/data1/ycb/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/data1/ycb/cache/huggingface/hub

export XDG_CACHE_HOME=/data1/ycb/cache/xdg
export NO_ALBUMENTATIONS_UPDATE=1

CUDA_VISIBLE_DEVICES=6,7 \
WANDB_DISABLED=true \
PYTHONUNBUFFERED=1 \
uv run --no-sync torchrun --nproc_per_node=2 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data1/ycb/datasets/pick_and_place_apple_right_230/lerobot_v2.1 \
    --embodiment-tag UNITREE_G1_UPPER_BODY \
    --modality-config-path examples/G1/upper_body/unitree_g1_upper_body_config.py \
    --num-gpus 1 \
    --output-dir /data1/ycb/checkpoints/GR00T_N1d7_30k_g1_22d_pick_and_place_apple_right_230/ \
    --max-steps 30000 \
    --save-steps 15000 \
    --global-batch-size 64 \
    --dataloader-num-workers 4 \
    --use-tensorboard
    # --resume-from-checkpoint