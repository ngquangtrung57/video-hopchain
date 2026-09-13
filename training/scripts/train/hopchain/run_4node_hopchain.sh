#!/bin/bash
# Wrapper for the four-node Video-HopChain runs: it sets the training
# environment, clears the nodes, checks that the GPUs are free and then calls
# the Ray launcher. launch_plain.sh and launch_swe.sh call it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO"

: "${HEAD_NODE:?set HEAD_NODE (trainer, Ray head)}"
: "${HEAD_JOB:?set HEAD_JOB to the full scheduler job id}"
: "${WORKER1_NODE:?set WORKER1_NODE (trainer)}"
: "${WORKER1_JOB:?set WORKER1_JOB}"
: "${WORKER2_NODE:?set WORKER2_NODE (rollout)}"
: "${WORKER2_JOB:?set WORKER2_JOB}"
: "${WORKER3_NODE:?set WORKER3_NODE (rollout)}"
: "${WORKER3_JOB:?set WORKER3_JOB}"
export HEAD_NODE HEAD_JOB WORKER1_NODE WORKER1_JOB WORKER2_NODE WORKER2_JOB WORKER3_NODE WORKER3_JOB
export NUM_WORKERS=3

if [ -n "${WORKER4_NODE:-}" ] || [ -n "${WORKER5_NODE:-}" ]; then
    echo "ERROR: WORKER4_NODE/WORKER5_NODE are set. This is the 4-node arm;" >&2
    echo "       it takes exactly one head node and three workers." >&2
    exit 12
fi

# Optional shell file and conda environment activated on every node before a
# remote command runs; leave unset if the login shell already provides them.
export CONFIG_FILE=${CONFIG_FILE:-}
export CONDA_ENV=${CONDA_ENV:-}

export TRAIN_SCRIPT=scripts/train/hopchain/grpo_video_4node_8b_hopchain.sh
export EXP_NAME_OVERRIDE=${EXP_NAME_OVERRIDE:-hopchain_grpo}
export TRAINER_NNODES=2
export ROLLOUT_NNODES=2

export TRAIN_PROMPT_MINI_BSZ=16

export TOTAL_EPOCHS=4

export CKPT_SAVE_CONTENTS="[model,optimizer,extra,hf_model]"
export SAVE_FREQ=10
export TEST_FREQ=10

export VAL_BEFORE_TRAIN=true

export VAL_MAX_SAMPLES=1000
: "${MODEL_PATH_OVERRIDE:?set MODEL_PATH_OVERRIDE to the merged stage-one checkpoint}"
export MODEL_PATH_OVERRIDE
export RESUME_MODE=auto

export NCCL_IB_DISABLE=0
# NCCL_IB_HCA is site specific. Export it before calling this script if the
# fabric needs an explicit device list.
export USE_FUSED_KERNELS=false
export FUSED_KERNEL_BACKEND=torch
export FORWARD_PREFETCH=true
export ACTOR_PPO_MAX_TOKEN_LEN=26004

export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN

export VERL_THINK_PREFILL=1
export VERL_GRADED_THINK_FORMAT=0

export ATTN_IMPL=${ATTN_IMPL:-flash_attention_3}
export RVRL_DECODE_THREADS=${RVRL_DECODE_THREADS:-1}
export RVRL_DECODE_CACHE=${RVRL_DECODE_CACHE:-8}
export RVRL_DECODE_EXECUTOR=${RVRL_DECODE_EXECUTOR:-1}

export EXPLORE_ENABLE=${EXPLORE_ENABLE:-false}
export SWE_ENABLE=${SWE_ENABLE:-false}

export TOP_PROB_THRESHOLD=${TOP_PROB_THRESHOLD:-0.95}
export EXPLORE_MAX_MEAN=${EXPLORE_MAX_MEAN:-1.0}
export TRIGGER_MODE=${TRIGGER_MODE:-high}
export TAU_LOW=${TAU_LOW:-0.8}
export TAU_HIGH=${TAU_HIGH:-0.95}
export SKIP_SPECIAL=${SKIP_SPECIAL:-true}
export SKIP_WS_PUNCT=${SKIP_WS_PUNCT:-true}
export SKIP_DIGITS=${SKIP_DIGITS:-true}
export SKIP_SUBWORD=${SKIP_SUBWORD:-false}
export VERL_EXPLORATION_IPC_DIR=${VERL_EXPLORATION_IPC_DIR:-/tmp/rvrl_drop_pos_${EXP_NAME_OVERRIDE}}

LOGDIR="$REPO/logs/prod_runs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
WRAP_LOG="$LOGDIR/run_4node_hopchain_${STAMP}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WRAP_LOG"; }

LOCK="$REPO/.run_4node_${EXP_NAME_OVERRIDE}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] REFUSING TO START: another wrapper holds $LOCK" | tee -a "$WRAP_LOG"
    echo "  holder: $(cat "$LOCK" 2>/dev/null)" | tee -a "$WRAP_LOG"
    exit 3
fi
echo "pid=$$ host=$(hostname) started=$(date '+%F %T')" >&9
log "acquired single-instance lock ($LOCK)"

ALL_NODES=(
    "${HEAD_NODE}:${HEAD_JOB}"
    "${WORKER1_NODE}:${WORKER1_JOB}"
    "${WORKER2_NODE}:${WORKER2_JOB}"
    "${WORKER3_NODE}:${WORKER3_JOB}"
)
MY_HOST=$(hostname)

remote_sh() {
    local node=$1 job=$2 cmd=$3
    PBS_JOBID="$job" timeout "${REMOTE_SH_TIMEOUT:-90}" \
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
            -o ServerAliveInterval=10 -o ServerAliveCountMax=3 "$node" "
        ${CONFIG_FILE:+source $CONFIG_FILE 2>/dev/null}
        ${CONDA_ENV:+conda activate $CONDA_ENV 2>/dev/null}
        cd $REPO
        $cmd
    " 9>&-
}

trap 'rc=$?; log "wrapper exiting rc=$rc"' EXIT

log "=== Step 1: cleaning all ${#ALL_NODES[@]} nodes ==="
KILL_BY_NAME="pkill -9 -u \$USER -f 'launch_pbs_4node[_]separate_jobs' 2>/dev/null; \
pkill -9 -u \$USER -f 'fully_async_mai[n]' 2>/dev/null; \
pkill -9 -u \$USER -f 'VLLM::Engine[C]ore' 2>/dev/null; true"

CLEAN_CMD="$KILL_BY_NAME; \
for i in 1 2 3; do \
$KILL_BY_NAME; \
sleep 4; \
done; \
timeout 30 ray stop --force >/dev/null 2>&1; \
rm -f /dev/shm/psm_* /dev/shm/nccl_* /dev/shm/torch_* 2>/dev/null; \
echo \"procs=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) mem=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB' | sort -rn | head -1)\""

clean_local() {
    for i in 1 2 3; do
        pkill -9 -u "$USER" -f 'launch_pbs_4node[_]separate_jobs' 2>/dev/null
        pkill -9 -u "$USER" -f 'fully_async_mai[n]' 2>/dev/null
        pkill -9 -u "$USER" -f 'VLLM::Engine[C]ore' 2>/dev/null
        sleep 4
    done
    timeout 30 ray stop --force >/dev/null 2>&1
    rm -f /dev/shm/psm_* /dev/shm/nccl_* /dev/shm/torch_* 2>/dev/null
}

for entry in "${ALL_NODES[@]}"; do
    node="${entry%%:*}"; job="${entry##*:}"
    if [ "$node" = "$MY_HOST" ]; then
        clean_local
        log "[$node] cleaned: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
    else
        log "[$node] cleaned: $(remote_sh "$node" "$job" "$CLEAN_CMD" 2>&1 | tail -1)"
    fi
done

log "=== Step 2: verifying GPUs are free (hard gate) ==="
PROBE="echo \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB' | sort -rn | head -1)\""
BUSY=0
for entry in "${ALL_NODES[@]}"; do
    node="${entry%%:*}"; job="${entry##*:}"
    R=""
    for attempt in 1 2 3; do
        if [ "$node" = "$MY_HOST" ]; then
            R=$(eval "$PROBE" 2>/dev/null)
        else
            R=$(remote_sh "$node" "$job" "$PROBE" 2>/dev/null | tail -1)
        fi
        if echo "$R" | grep -qE '^[0-9]+ [0-9]+$'; then break; fi
        log "[$node] probe attempt $attempt returned no usable data; retrying"
        R=""
        sleep 10
    done
    if [ -z "$R" ]; then
        log "[$node] ERROR: GPU probe failed 3x (ssh unreachable?) - refusing to launch"
        BUSY=1
        continue
    fi
    NPROC=$(echo "$R" | awk '{print $1}'); NPROC=${NPROC:-99}
    USED=$(echo "$R" | awk '{print $2}');  USED=${USED:-999999}
    log "[$node] gpu_procs=$NPROC  max_mem=${USED} MiB"
    if [ "$NPROC" -ne 0 ] 2>/dev/null || [ "$USED" -gt 500 ] 2>/dev/null; then
        BUSY=1; log "[$node] ERROR: GPUs still occupied"
    fi
done
if [ "$BUSY" = "1" ]; then
    log "ABORT: GPUs not free on at least one node - refusing to launch."
    exit 4
fi
log "all ${#ALL_NODES[@]} nodes clean"

INIT_MODEL="$MODEL_PATH_OVERRIDE"
if [ ! -f "$INIT_MODEL/config.json" ] || ! ls "$INIT_MODEL"/*.safetensors >/dev/null 2>&1; then
    log "ABORT: init model missing or unmerged at $INIT_MODEL"
    exit 5
fi
NSAFE=$(ls -1 "$INIT_MODEL"/*.safetensors 2>/dev/null | wc -l)
log "init model OK: $INIT_MODEL ($NSAFE safetensors shards)"

log "============================================"
log "Launching 4-node training (2 trainer + 2 rollout)"
log "  exp:        $EXP_NAME_OVERRIDE"
log "  topology:   ${TRAINER_NNODES} trainer (dp=$((TRAINER_NNODES * 8))) + ${ROLLOUT_NNODES} rollout"
log "  mini_bsz:   $TRAIN_PROMPT_MINI_BSZ prompts (x8 resp = $((TRAIN_PROMPT_MINI_BSZ * 8)) seq; % dp=$((TRAINER_NNODES*8)) -> $(( (TRAIN_PROMPT_MINI_BSZ*8) % (TRAINER_NNODES*8) )))"
log "  init:       WARM $MODEL_PATH_OVERRIDE (resume_mode=$RESUME_MODE)"
log "  explore:    enable=$EXPLORE_ENABLE swe=$SWE_ENABLE"
log "  val_max:    $VAL_MAX_SAMPLES"
log "  trigger:    mode=$TRIGGER_MODE tau=$TOP_PROB_THRESHOLD band=[$TAU_LOW,$TAU_HIGH]"
log "  filter:     special=$SKIP_SPECIAL ws_punct=$SKIP_WS_PUNCT digits=$SKIP_DIGITS subword_cont=$SKIP_SUBWORD"
log "  mask_ipc:   $VERL_EXPLORATION_IPC_DIR"
log "  epochs:     $TOTAL_EPOCHS (cumulative)  save_freq=$SAVE_FREQ  ckpt=$CKPT_SAVE_CONTENTS"
log "  format:     VERL_THINK_PREFILL=$VERL_THINK_PREFILL  VERL_GRADED_THINK_FORMAT=$VERL_GRADED_THINK_FORMAT"
log "  perf:       ATTN_IMPL=$ATTN_IMPL decode(threads=$RVRL_DECODE_THREADS cache=$RVRL_DECODE_CACHE executor=$RVRL_DECODE_EXECUTOR) val_before_train=$VAL_BEFORE_TRAIN"
log "============================================"

bash scripts/launch_pbs_4node_separate_jobs.sh full 9>&- 2>&1 | tee -a "$WRAP_LOG"
TRAIN_RC=${PIPESTATUS[0]}

log "Training returned rc=$TRAIN_RC"

exit "$TRAIN_RC"
