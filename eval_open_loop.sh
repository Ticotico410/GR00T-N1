#!/bin/bash
set -euo pipefail

# Local open-loop eval. Cosmos = local NVMe HF cache (not /sh/ycb training path).
cd /home/karthus_chen/ycb_ws/GR00T-N1
source .venv/bin/activate
# Drop any leftover Cosmos env from the parent shell; local_inference_env.sh sets NVMe path.
unset COSMOS_REASON2_PATH LOCAL_COSMOS_REASON2_PATH
# shellcheck source=scripts/local_inference_env.sh
source scripts/local_inference_env.sh

CKPT=/home/karthus_chen/unitree_sh_disk/tools/ycb/checkpoints/GR00T_N1d7_80k_g1_wbc_pick_up_multiple_cushions_brainco_200/checkpoint-50000
OUT_DIR=/home/karthus_chen/ycb_ws/GR00T-N1/eval/GR00T_N1d7_80k_g1_wbc_pick_up_multiple_cushions_brainco_200/open_loop_eval
mkdir -p "${OUT_DIR}"

# absolute (default, original): full action concat in decoded absolute space
# relative9d: robot_root only in local-xyz + rot6d (training target space)
ROOT_EVAL_SPACE="absolute"

if [[ "${ROOT_EVAL_SPACE}" == "relative9d" ]]; then
  SAVE_PLOT="${OUT_DIR}/traj_relative9d.jpeg"
else
  SAVE_PLOT="${OUT_DIR}/traj_absolute.jpeg"
fi

python gr00t/eval/open_loop_eval.py \
  --dataset-path /home/karthus_chen/unitree_sh_disk/demo_data_June/livingroom/pick_up_multiple_cushions_brainco/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_WBC \
  --model-path "${CKPT}" \
  --save_plot_path "${SAVE_PLOT}" \
  --traj-ids 150 \
  --denoising-steps 4 \
  --action-horizon 48 \
  --steps 800 \
  --video-backend pyav \
  --root-eval-space "${ROOT_EVAL_SPACE}" \
  --ema-alpha 0.25
