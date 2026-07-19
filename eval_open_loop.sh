#!/bin/bash
set -euo pipefail

source /sh/ycb/venvs/gr00t_n1d7/bin/activate
cd /sh/ycb/model/GR00T
export PYTHONPATH=/sh/ycb/model/GR00T:${PYTHONPATH:-}
export HF_HOME=/sh/ycb/.cache/huggingface
export HF_HUB_CACHE=/sh/ycb/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1

CKPT=/sh/ycb/checkpoints/GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200/GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200/checkpoint-40000
OUT_DIR=/sh/ycb/checkpoints/GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200/open_loop_eval
mkdir -p "${OUT_DIR}"

# absolute (default, original): full action concat in decoded absolute space
# relative9d: robot_root only in local-xyz + rot6d (training target space)
ROOT_EVAL_SPACE="relative9d"

if [[ "${ROOT_EVAL_SPACE}" == "relative9d" ]]; then
  SAVE_PLOT="${OUT_DIR}/traj_relative9d.jpeg"
else
  SAVE_PLOT="${OUT_DIR}/traj_absolute.jpeg"
fi

python gr00t/eval/open_loop_eval.py \
  --dataset-path /sh/datasets/g1/pick_up_multiple_cushions_brainco_200/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_WBC \
  --model-path "${CKPT}" \
  --save_plot_path "${SAVE_PLOT}" \
  --traj-ids 150 \
  --denoising-steps 4 \
  --action-horizon 48 \
  --steps 800 \
  --video-backend pyav \
  --root-eval-space "${ROOT_EVAL_SPACE}"
