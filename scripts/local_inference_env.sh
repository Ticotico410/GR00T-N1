# Shared env for local open-loop inference.
# Training (train.sh) keeps server Cosmos under /sh/ycb — do not source this there.
#
# Usage (from repo root):
#   source scripts/local_inference_env.sh
#
# Override CosmOS location if needed:
#   LOCAL_COSMOS_REASON2_PATH=/other/snapshot source scripts/local_inference_env.sh

GR00T_ROOT="${GR00T_ROOT:-/home/karthus_chen/ycb_ws/GR00T-N1}"
LOCAL_HF_HOME="${LOCAL_HF_HOME:-/home/karthus_chen/.cache/huggingface}"
# Hard default: local NVMe HF snapshot (not NFS unitree_sh_disk, not /sh/ycb).
_DEFAULT_LOCAL_COSMOS="${LOCAL_HF_HOME}/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
LOCAL_COSMOS_REASON2_PATH="${LOCAL_COSMOS_REASON2_PATH:-${_DEFAULT_LOCAL_COSMOS}}"
unset _DEFAULT_LOCAL_COSMOS

export PYTHONPATH="${GR00T_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${LOCAL_HF_HOME}"
export HF_HUB_CACHE="${LOCAL_HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GROOT_PATCH_MISTRAL="${GROOT_PATCH_MISTRAL:-1}"
export GROOT_HF_LOCAL_FIRST="${GROOT_HF_LOCAL_FIRST:-1}"

# Force inference CosmOS to the local path. Ignore any parent-shell
# COSMOS_REASON2_PATH leftover (e.g. NFS or /sh/ycb).
export COSMOS_REASON2_PATH="${LOCAL_COSMOS_REASON2_PATH}"

if [[ ! -d "${COSMOS_REASON2_PATH}" ]]; then
  echo "ERROR: local Cosmos path missing: ${COSMOS_REASON2_PATH}" >&2
  echo "Set LOCAL_COSMOS_REASON2_PATH to your local Cosmos-Reason2-2B snapshot." >&2
  return 1 2>/dev/null || exit 1
fi

echo "[local_inference_env] HF_HOME=${HF_HOME}"
echo "[local_inference_env] COSMOS_REASON2_PATH=${COSMOS_REASON2_PATH}"
