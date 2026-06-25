#!/bin/bash
set -e

source /mnt/unitree_cpfs/wangcong/anaconda3/etc/profile.d/conda.sh
conda activate gr00t_n1

cd /root/shanghai/ycb/GR00T-N1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export MASTER_PORT=29531

echo "===== ENV CHECK ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MASTER_PORT=${MASTER_PORT}"

python -u scripts/gr00t_finetune.py \
  --dataset-path /data1/ycb/datasets/pick_and_place_apple_right_230/lerobot_v2.1 \
  --num-gpus 2 \
  --output-dir /data1/ycb/checkpoints/GR00T_N1_40k_g1_22d_pick_and_place_apple_right_230/  \
  --max-steps 40000 \
  --save-steps 20000 \
  --batch-size 16 \
  --data-config unitree_g1_upper_body \
  --video-backend torchvision_av \
  --report-to tensorboard