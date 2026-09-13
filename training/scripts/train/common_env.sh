#!/bin/bash
# Shared environment for the training launchers: repository root, scratch
# location, CUDA paths and the logger list.

_COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLVR_ROOT="$(cd "${_COMMON_ENV_DIR}/../.." && pwd)"

LOCAL_ENV="${RLVR_ROOT}/local_env.sh"
if [ -f "$LOCAL_ENV" ]; then
    source "$LOCAL_ENV"
fi

ENV_FILE="${RLVR_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

if [ -z "$SCRATCH_DIR" ]; then
    echo "ERROR: SCRATCH_DIR not set. Create local_env.sh or export it." >&2
    echo "  Example: export SCRATCH_DIR=/path/to/scratch" >&2
    exit 1
fi

export OUTPUT_BASE=${SCRATCH_DIR}/checkpoints

if [ -z "$CUDA_HOME" ]; then
    if command -v nvcc &>/dev/null; then
        export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
    fi
fi

if [ -n "$CUDA_HOME" ]; then
    export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
fi

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VERL_WEIGHT_TRANSFER_SHM=1
export RAY_DASHBOARD_OFF=${RAY_DASHBOARD_OFF:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if python3 -c "import wandb" 2>/dev/null; then
    export TRAINER_LOGGER='["console","wandb"]'
else
    export TRAINER_LOGGER='["console"]'
fi
