#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

N1_UV_PYTHON="/root/shanghai/ycb/GR00T-N1/.venv/bin/python"

mkdir -p /data1/ycb/cache/huggingface/hub
mkdir -p /data1/ycb/cache/huggingface/datasets
mkdir -p /data1/ycb/cache/xdg

export HF_HOME=/data1/ycb/cache/huggingface
export HF_HUB_CACHE=/data1/ycb/cache/huggingface/hub
export HF_DATASETS_CACHE=/data1/ycb/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/data1/ycb/cache/huggingface/hub

export XDG_CACHE_HOME=/data1/ycb/cache/xdg
export NO_ALBUMENTATIONS_UPDATE=1

export CUDA_VISIBLE_DEVICES=2,3
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

echo "===== ENV CHECK ====="
echo "PWD=$(pwd)"
echo "PYTHON=${N1_UV_PYTHON}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

"${N1_UV_PYTHON}" -c "import sys, torch; print(sys.executable); print('torch', torch.__version__)"
"${N1_UV_PYTHON}" -c "from gr00t.configs.base_config import get_default_config; print('import gr00t ok')"

"${N1_UV_PYTHON}" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29532 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data1/ycb/datasets/pick_and_place_apple_right_230/lerobot_v2.1 \
    --embodiment-tag UNITREE_G1_UPPER_RIGHT_HAND \
    --modality-config-path examples/G1/upper_right_hand/unitree_g1_upper_right_hand_config.py \
    --num-gpus 2 \
    --output-dir /data1/ycb/checkpoints/GR00T_N1d7_40k_g1_22d_pick_and_place_apple_right_230/ \
    --max-steps 40000 \
    --save-steps 20000 \
    --global-batch-size 64 \
    --dataloader-num-workers 4 \
    --use-tensorboard