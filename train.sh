#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p /data1/ycb/cache/huggingface/hub
mkdir -p /data1/ycb/cache/huggingface/datasets
mkdir -p /data1/ycb/cache/xdg

export HF_HOME=/data1/ycb/cache/huggingface
export HF_HUB_CACHE=/data1/ycb/cache/huggingface/hub
export HF_DATASETS_CACHE=/data1/ycb/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/data1/ycb/cache/huggingface/hub

export XDG_CACHE_HOME=/data1/ycb/cache/xdg
export NO_ALBUMENTATIONS_UPDATE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_DISABLED=true \
PYTHONUNBUFFERED=1 \
uv run torchrun --nproc_per_node=4 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data1/ycb/datasets/pick_and_place_apple_bidirectional_brainco_60/lerobot_v2.1 \
    --embodiment-tag UNITREE_G1_UPPER_BODY \
    --modality-config-path examples/G1/upper_body/unitree_g1_upper_body_config.py \
    --num-gpus 4 \
    --output-dir /data1/ycb/checkpoints/gr00t_n1d7_36_brainco_pick_and_place_apple_bidirectional_brainco_60_50k \
    --max-steps 50000 \
    --save-steps 25000 \
    --global-batch-size 128 \
    --dataloader-num-workers 4
    # --resume-from-checkpoint