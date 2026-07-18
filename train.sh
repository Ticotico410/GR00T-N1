#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/sh/ycb/model/GR00T"
VENV_PYTHON="/sh/ycb/venvs/gr00t_n1d7/bin/activate"
CACHE_ROOT="/sh/ycb/.cache"
DATASET_ROOT="/sh/datasets/g1/pick_up_multiple_cushions_brainco_200/lerobot_v2.1"

EXP_NAME="GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200"
CHECKPOINT_BASE_DIR="/sh/ycb/checkpoints"
OUTPUT_DIR="${CHECKPOINT_BASE_DIR}/${EXP_NAME}"
BASE_MODEL_PATH="nvidia/GR00T-N1.7-3B"
EMBODIMENT_TAG="UNITREE_G1_WBC"
MODALITY_CONFIG_PATH="examples/G1/wbc/unitree_g1_wbc_config.py"

NUM_GPUS=2
CUDA_DEVICES="0,1"
GLOBAL_BATCH_SIZE=64
MAX_STEPS=60000
SAVE_STEPS=10000
DATALOADER_NUM_WORKERS=4
WANDB_PROJECT="GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${CACHE_ROOT}/huggingface/hub"
mkdir -p "${CACHE_ROOT}/huggingface/datasets"
mkdir -p "${CACHE_ROOT}/xdg"
mkdir -p "${CACHE_ROOT}/uv"
mkdir -p "${CHECKPOINT_BASE_DIR}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"

export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONUNBUFFERED=1

echo "===== ENV CHECK ====="
echo "PWD=$(pwd)"
echo "PYTHON=${VENV_PYTHON}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

"${VENV_PYTHON}" -c "import sys, torch; print(sys.executable); print('torch', torch.__version__)"
"${VENV_PYTHON}" -c "from gr00t.configs.base_config import get_default_config; print('import gr00t ok')"

"${VENV_PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${DATASET_ROOT}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --modality-config-path "${MODALITY_CONFIG_PATH}" \
    --num-gpus "${NUM_GPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name "${EXP_NAME}" \
    --max-steps "${MAX_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --use-wandb
