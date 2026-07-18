#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/sh/ycb/model/GR00T"
VENV_PATH="/sh/ycb/venvs/gr00t_n1d7/bin/activate"
CACHE_ROOT="/sh/ycb/.cache"
DATASET_ROOT="/sh/datasets/g1/pick_up_multiple_cushions_brainco_200/lerobot_v2.1"

EXP_NAME="GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200"
CHECKPOINT_BASE_DIR="/sh/ycb/checkpoints"
OUTPUT_DIR="${CHECKPOINT_BASE_DIR}/${EXP_NAME}"
BASE_MODEL_PATH="${CACHE_ROOT}/gr00t_n1d7/GR00T-N1.7-3B"
EMBODIMENT_TAG="UNITREE_G1_WBC"
MODALITY_CONFIG_PATH="examples/G1/wbc/unitree_g1_wbc_config.py"

NUM_GPUS=2
CUDA_DEVICES="0,1"
GLOBAL_BATCH_SIZE=64
MAX_STEPS=80000
SAVE_STEPS=10000
DATALOADER_NUM_WORKERS=4
WANDB_PROJECT="GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200"

# W&B 开关：tyro 布尔要用 --use-wandb / --no-use-wandb，不能写 =true/=false
WANDB_ENABLED=true

cd "${PROJECT_ROOT}"
source "${VENV_PATH}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 所有可变缓存放到 /sh/ycb/.cache
mkdir -p "${CACHE_ROOT}/huggingface/hub"
mkdir -p "${CACHE_ROOT}/huggingface/datasets"
mkdir -p "${CACHE_ROOT}/xdg"
mkdir -p "${CACHE_ROOT}/uv"
mkdir -p "${CACHE_ROOT}/torch"
mkdir -p "${CACHE_ROOT}/torch_extensions"
mkdir -p "${CACHE_ROOT}/triton"
mkdir -p "${CACHE_ROOT}/cuda"
mkdir -p "${CACHE_ROOT}/tmp"
mkdir -p "${CACHE_ROOT}/wandb"
mkdir -p "${CHECKPOINT_BASE_DIR}"

# Hugging Face / transformers（Cosmos-Reason2 等会走这里）
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# 通用 / 构建类缓存（默认会落到 /root/.cache）
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${CACHE_ROOT}/tmp"
export TMP="${CACHE_ROOT}/tmp"

# wandb 本地目录也放到大盘（cwd 下的 ./wandb 仍会在 PROJECT_ROOT，这里管 cache/config）
export WANDB_DIR="${OUTPUT_DIR}"
export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb/cache"
export WANDB_CONFIG_DIR="${CACHE_ROOT}/wandb/config"
export WANDB_DATA_DIR="${CACHE_ROOT}/wandb/data"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}" "${WANDB_DIR}"

export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONUNBUFFERED=1

# 远程机没有本地 Clash；错误代理会导致 wandb ProxyError
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset WANDB_DISABLED
unset WANDB_MODE

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: 请先: export WANDB_API_KEY='你的key'  （https://wandb.ai/authorize）"
    exit 1
  fi
  export WANDB_API_KEY
  WANDB_FLAG=(--use-wandb)
else
  export WANDB_DISABLED=true
  export WANDB_MODE=disabled
  WANDB_FLAG=(--no-use-wandb)
fi

# 若只在 /root/.cache 登过 HF，把 token 同步到 HF_HOME，避免 gated 模型 401
if [[ ! -f "${HF_HOME}/token" && -f /root/.cache/huggingface/token ]]; then
  cp /root/.cache/huggingface/token "${HF_HOME}/token"
  echo "Synced HF token -> ${HF_HOME}/token"
fi

echo "===== ENV CHECK ====="
echo "PWD=$(pwd)"
echo "VENV=${VENV_PATH}"
echo "PYTHON=$(command -v python)"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "TMPDIR=${TMPDIR}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "WANDB_ENABLED=${WANDB_ENABLED}"
echo "WANDB_DIR=${WANDB_DIR}"

python -m torch.distributed.run \
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
  "${WANDB_FLAG[@]}"
