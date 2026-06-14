#!/bin/bash

source /mnt/unitree_cpfs/wangcong/anaconda3/etc/profile.d/conda.sh
conda activate gr00t_n1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_DISABLED=true \
PYTHONUNBUFFERED=1 \
python -u scripts/gr00t_finetune.py \
  --dataset-path /data1/ycb/datasets/supermarket_shelf_organizing_brainco_201 \
  --num-gpus 4 \
  --output-dir /data1/ycb/checkpoints/gr00t_n1_supermarket_shelf_organizing_brainco_201_60k \
  --max-steps 60000 \
  --save-steps 20000 \
  --batch-size 32 \
  --data-config unitree_g1_wbc \
  --video-backend torchvision_av \
  --report-to tensorboard