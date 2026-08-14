#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/sh/ycb/model/GR00T"
VENV_PATH="/sh/ycb/venvs/gr00t_n1d7/bin/activate"
CACHE_ROOT="/sh/ycb/.cache"
DATASET_ROOT="/sh/datasets/g1/smpl/tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/lerobot_v2.1"
EXP_NAME="GR00T_N1d7_100k_g1_smpl_euler_delta_tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco"
CHECKPOINT_BASE_DIR="/sh/ycb/checkpoints"
OUTPUT_DIR="${CHECKPOINT_BASE_DIR}/${EXP_NAME}"
BASE_MODEL_PATH="${CACHE_ROOT}/gr00t_n1d7/GR00T-N1.7-3B"
EMBODIMENT_TAG="UNITREE_G1_SMPL"
MODALITY_CONFIG_PATH="examples/G1/smpl/unitree_g1_smpl_config.py"

NUM_GPUS=2
CUDA_DEVICES="0,1"
GLOBAL_BATCH_SIZE=64
MAX_STEPS=100000
SAVE_STEPS=10000
LEARNING_RATE=1e-4
DATALOADER_NUM_WORKERS=1
WANDB_PROJECT="GR00T_N1d7_100k_g1_smpl_euler_delta_tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco"

# Init
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-0}"
# Resume
# export WANDB_RUN_ID=ahfu3a6i
# export WANDB_RESUME=allow
# RESUME_FROM_CHECKPOINT=1

# SMPL root training (see gr00t/configs/smpl_root_mode.py):
#   original     — 82D hip quat absolute; state drops robot_root (default)
#   rot6d        — 84D hip rot6d; --action-mode relative only
#   delta_euler  — 81D Δeuler vs state; fixed relative
#   euler        — 81D hip euler; --action-mode absolute|relative
# CLI: bash train.sh --root-process-mode rot6d --action-mode relative
# Legacy env (when ROOT_PROCESS_MODE unset): USE_RELATIVE_EULER / USE_STATE_EULER
ROOT_PROCESS_MODE="${ROOT_PROCESS_MODE:-}"
USE_RELATIVE_EULER="${USE_RELATIVE_EULER:-0}"
USE_STATE_EULER="${USE_STATE_EULER:-0}"
ACTION_MODE="${ACTION_MODE:-}"

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
# Training Cosmos MUST stay under /sh/ycb (server). Do not point at laptop
# ~/.cache/... — that path is only for local open-loop inference.
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
# 权重已在 HF_HUB_CACHE 时默认离线，避免再连 huggingface.co（adapter_config 等 HEAD）
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# transformers _patch_mistral_regex 会对 Hub ID 无条件调 model_info；Cosmos 非 mistral，跳过即可
export GROOT_PATCH_MISTRAL="${GROOT_PATCH_MISTRAL:-1}"
export GROOT_HF_LOCAL_FIRST="${GROOT_HF_LOCAL_FIRST:-1}"

# Pin Cosmos backbone to server HF snapshot when present.
_SERVER_COSMOS_SNAPS="${HF_HUB_CACHE}/models--nvidia--Cosmos-Reason2-2B/snapshots"
if [[ -z "${COSMOS_REASON2_PATH:-}" && -d "${_SERVER_COSMOS_SNAPS}" ]]; then
  COSMOS_REASON2_PATH="$(
    find "${_SERVER_COSMOS_SNAPS}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
  )"
  export COSMOS_REASON2_PATH
fi
unset _SERVER_COSMOS_SNAPS

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

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: 请先: export WANDB_API_KEY='你的key'  （https://wandb.ai/authorize）"
  exit 1
fi
export WANDB_API_KEY
# 默认开新 wandb run；仅当显式传入 WANDB_RUN_ID 时才 resume
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  export WANDB_RUN_ID
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
else
  unset WANDB_RUN_ID
  unset WANDB_RESUME
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
echo "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
echo "MODALITY_CONFIG_PATH=${MODALITY_CONFIG_PATH}"
echo "HF_HOME=${HF_HOME}"
echo "COSMOS_REASON2_PATH=${COSMOS_REASON2_PATH:-<unset; launch_finetune will scan HF_HUB_CACHE>}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "TMPDIR=${TMPDIR}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "WANDB_RUN_ID=${WANDB_RUN_ID:-<new run>}"
echo "WANDB_RESUME=${WANDB_RESUME:-<unset>}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "MAX_STEPS=${MAX_STEPS}"

RESUME_ARGS=()
if [[ "${RESUME_FROM_CHECKPOINT}" == "1" || "${RESUME_FROM_CHECKPOINT}" == "true" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint)
fi

ROOT_ARGS=()
if [[ -n "${ROOT_PROCESS_MODE}" ]]; then
  ROOT_ARGS+=(--root-process-mode "${ROOT_PROCESS_MODE}")
fi
if [[ -n "${ACTION_MODE}" ]]; then
  ROOT_ARGS+=(--action-mode "${ACTION_MODE}")
fi

LEGACY_ARGS=()
if [[ -z "${ROOT_PROCESS_MODE}" ]]; then
  if [[ "${USE_RELATIVE_EULER}" == "1" || "${USE_RELATIVE_EULER}" == "true" ]]; then
    LEGACY_ARGS+=(--use-relative-euler)
  fi
  if [[ "${USE_STATE_EULER}" == "1" || "${USE_STATE_EULER}" == "true" ]]; then
    LEGACY_ARGS+=(--use-state-euler)
  fi
fi

echo "ROOT_PROCESS_MODE=${ROOT_PROCESS_MODE:-original (default)}"
echo "USE_RELATIVE_EULER=${USE_RELATIVE_EULER}"
echo "USE_STATE_EULER=${USE_STATE_EULER}"
echo "ACTION_MODE=${ACTION_MODE:-<unset>}"
echo "Extra CLI args: $*"

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
  --learning-rate "${LEARNING_RATE}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --wandb-project "${WANDB_PROJECT}" \
  "${RESUME_ARGS[@]}" \
  "${ROOT_ARGS[@]}" \
  "${LEGACY_ARGS[@]}" \
  --use-wandb \
  "$@"
