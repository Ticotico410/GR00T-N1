#!/bin/bash
set -euo pipefail

# Action-head cross-attention confusion matrices (action steps x VL tokens).
cd /home/karthus_chen/ycb_ws/GR00T-N1
source .venv/bin/activate
unset COSMOS_REASON2_PATH LOCAL_COSMOS_REASON2_PATH
# shellcheck source=scripts/local_inference_env.sh
source scripts/local_inference_env.sh

CKPT=/home/karthus_chen/unitree_sh_disk/tools/ycb/checkpoints/GR00T_N1d7_100k_g1_wbc_pick_up_multiple_cushions_brainco_200/checkpoint-100000
OUT_DIR=/home/karthus_chen/ycb_ws/GR00T-N1/eval/GR00T_N1d7_100k_g1_wbc_pick_up_multiple_cushions_brainco_200/action_vl_eval
mkdir -p "${OUT_DIR}"

python gr00t/eval/action_vl_eval.py \
  --model-path "${CKPT}" \
  --dataset-path /home/karthus_chen/unitree_sh_disk/demo_data_June/livingroom/pick_up_multiple_cushions_brainco/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_WBC \
  --traj-ids 150 \
  --frame 200 \
  --task "Pick up all scattered cushions and gather them together olderly." \
  --save-dir "${OUT_DIR}" \
  --video-backend pyav \
  --action-stride 4
