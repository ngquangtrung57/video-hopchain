#!/bin/bash
# Start a Ray cluster across the head node and its workers, verify the GPU
# count, then run the training script on it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${HEAD_NODE:?set HEAD_NODE to the hostname of the head node}"
: "${HEAD_JOB:?set HEAD_JOB to the scheduler job id of the head node}"

WORKER1_NODE=${WORKER1_NODE:-}
WORKER1_JOB=${WORKER1_JOB:-}

WORKER2_NODE=${WORKER2_NODE:-}
WORKER2_JOB=${WORKER2_JOB:-}

WORKER3_NODE=${WORKER3_NODE:-}
WORKER3_JOB=${WORKER3_JOB:-}

WORKER4_NODE=${WORKER4_NODE:-}
WORKER4_JOB=${WORKER4_JOB:-}
WORKER5_NODE=${WORKER5_NODE:-}
WORKER5_JOB=${WORKER5_JOB:-}
NUM_WORKERS=${NUM_WORKERS:-3}
TOTAL_NODES=$((NUM_WORKERS + 1))
EXPECT_GPUS=$((TOTAL_NODES * 8))

RAY_PORT=${RAY_PORT:-6379}
# Optional shell file and conda environment activated on every node before the
# job starts; leave unset if the login shell already provides them.
CONFIG_FILE=${CONFIG_FILE:-}
CONDA_ENV=${CONDA_ENV:-}
CUDA_HOME_PATH=${CUDA_HOME_PATH:-}
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/hopchain/grpo_video_4node_8b_hopchain.sh}"
LOG_DIR="${REPO_DIR}/logs/prod_runs"
RAY_TMPDIR=/tmp/ray_${USER}

RUN_MODE=${1:-smoke}
case "$RUN_MODE" in
    smoke|full) ;;
    *) echo "ERROR: arg 1 must be 'smoke' or 'full', got '$RUN_MODE'"; exit 1 ;;
esac
export RUN_MODE

mkdir -p "$LOG_DIR"

echo "============================================"
echo "${TOTAL_NODES}-NODE LAUNCHER  RUN_MODE=$RUN_MODE"
echo "Started:   $(date)"
echo "Head:    $HEAD_NODE (job $HEAD_JOB)  [LOCAL]"
for N in $(seq 1 "$NUM_WORKERS"); do
    NODE_VAR="WORKER${N}_NODE"; JOB_VAR="WORKER${N}_JOB"
    if [ -z "${!NODE_VAR}" ] || [ -z "${!JOB_VAR}" ]; then
        echo "ERROR: NUM_WORKERS=$NUM_WORKERS but $NODE_VAR or $JOB_VAR is empty" >&2
        exit 1
    fi
    echo "Worker$N: ${!NODE_VAR} (job ${!JOB_VAR})"
done
echo "Train:   $TRAIN_SCRIPT"
echo "Ray temp dir: $RAY_TMPDIR"
echo "============================================"

HOST=$(hostname)
if [ "$HOST" != "$HEAD_NODE" ]; then
    echo "ERROR: Must run on HEAD node ($HEAD_NODE), currently on $HOST" >&2
    exit 1
fi

ssh_remote() {
    local node=$1
    local job=$2
    local cmd=$3
    PBS_JOBID="$job" ssh -o StrictHostKeyChecking=no "$node" "
        ${CONFIG_FILE:+source $CONFIG_FILE}
        ${CONDA_ENV:+conda activate $CONDA_ENV}
        ${CUDA_HOME_PATH:+export CUDA_HOME=$CUDA_HOME_PATH}
        ${CUDA_HOME_PATH:+export CUDA_PATH=$CUDA_HOME_PATH}
        export VERL_WEIGHT_TRANSFER_SHM=1
        export RAY_memory_monitor_refresh_ms=0
        export RAY_memory_usage_threshold=0.98
        export VLLM_ALLREDUCE_USE_SYMM_MEM=0
        export GLIBC_TUNABLES=\${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2048}
        export VERL_GRADED_THINK_FORMAT=${VERL_GRADED_THINK_FORMAT:-0}
        export VERL_THINK_PREFILL=${VERL_THINK_PREFILL:-0}
        export PYTHONPATH=${REPO_DIR}/rewards:\${PYTHONPATH:-}
        export VERL_MAX_CONCURRENT_PER_REPLICA=${VERL_MAX_CONCURRENT_PER_REPLICA:-16}
        export NVCC_APPEND_FLAGS='-ccbin /usr/bin/g++'
        export VLLM_USE_FLASHINFER_SAMPLER=0
        export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
        ${NCCL_IB_HCA:+export NCCL_IB_HCA=$NCCL_IB_HCA}
        export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
        export RVRL_DECODE_THREADS=${RVRL_DECODE_THREADS:-0}
        export RVRL_DECODE_CACHE=${RVRL_DECODE_CACHE:-0}
        export RVRL_DECODE_EXECUTOR=${RVRL_DECODE_EXECUTOR:-0}
        $cmd
    "
}

echo ""
echo "=== Step 1: Cleaning up stale Ray on all nodes ==="
echo "[$HEAD_NODE] (local) cleaning..."
ray stop --force 2>/dev/null || true
pkill -9 -f 'VLLM::Engine[C]ore' 2>/dev/null || true
pkill -9 -f 'fully_async_main' 2>/dev/null || true
sleep 2

for N in $(seq 1 "$NUM_WORKERS"); do
    NODE_VAR="WORKER${N}_NODE"
    JOB_VAR="WORKER${N}_JOB"
    NODE=${!NODE_VAR}
    JOB=${!JOB_VAR}
    echo "[$NODE] cleaning..."
    ssh_remote "$NODE" "$JOB" "ray stop --force 2>/dev/null || true; pkill -9 -f 'VLLM::Engine[C]ore' 2>/dev/null || true; pkill -9 -f 'fully_async_main' 2>/dev/null || true; sleep 2"
done

echo ""
echo "=== Step 2: Starting Ray head on $HEAD_NODE ==="
HEAD_IP=$(hostname -i | awk '{print $1}')
echo "Head IP: $HEAD_IP:$RAY_PORT"

[ -n "$CONFIG_FILE" ] && source "$CONFIG_FILE"
[ -n "$CONDA_ENV" ] && conda activate "$CONDA_ENV"
if [ -n "$CUDA_HOME_PATH" ]; then
    export CUDA_HOME=$CUDA_HOME_PATH
    export CUDA_PATH=$CUDA_HOME_PATH
fi
export VERL_WEIGHT_TRANSFER_SHM=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.98
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VERL_GRADED_THINK_FORMAT=${VERL_GRADED_THINK_FORMAT:-0}
export VERL_THINK_PREFILL=${VERL_THINK_PREFILL:-0}
export VERL_MAX_CONCURRENT_PER_REPLICA=${VERL_MAX_CONCURRENT_PER_REPLICA:-16}
export NVCC_APPEND_FLAGS='-ccbin /usr/bin/g++'
export VLLM_USE_FLASHINFER_SAMPLER=0
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export GLIBC_TUNABLES=${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2048}

export PYTHONPATH=${REPO_DIR}/rewards:${PYTHONPATH:-}

ray start --head --port=$RAY_PORT --num-gpus=8 --num-cpus=112 \
    --temp-dir="$RAY_TMPDIR" --dashboard-host=0.0.0.0
sleep 5

echo ""
echo "=== Step 3: Starting Ray workers ==="
for N in $(seq 1 "$NUM_WORKERS"); do
    NODE_VAR="WORKER${N}_NODE"
    JOB_VAR="WORKER${N}_JOB"
    NODE=${!NODE_VAR}
    JOB=${!JOB_VAR}
    echo "[$NODE] starting worker..."
    ssh_remote "$NODE" "$JOB" "ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=8 --num-cpus=112 --temp-dir=$RAY_TMPDIR"
    sleep 3
done

echo ""
echo "=== Step 4: Verifying Ray cluster (expect $EXPECT_GPUS GPUs, $TOTAL_NODES nodes) ==="
python3 -c "
import ray
ray.init(address='$HEAD_IP:$RAY_PORT')
res = ray.cluster_resources()
gpus = int(res.get('GPU', 0))
nodes = sum(1 for k in res if k.startswith('node:'))
print(f'  GPUs:  {gpus}')
print(f'  Nodes: {nodes}')
print(f'  CPUs:  {int(res.get(\"CPU\", 0))}')
ray.shutdown()
assert gpus >= $EXPECT_GPUS, f'Expected >=$EXPECT_GPUS GPUs, got {gpus}'
assert nodes >= $TOTAL_NODES, f'Expected >=$TOTAL_NODES nodes, got {nodes}'
print('  Cluster OK')
"

echo ""
echo "=== Step 5: Launching training (RUN_MODE=$RUN_MODE) ==="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/grpo_video_4node_${RUN_MODE}_${TIMESTAMP}.log"
echo "Training log: $LOG_FILE"

export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
export RUN_MODE
cd "$REPO_DIR"
bash "$TRAIN_SCRIPT" 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "Training finished: $(date)"
echo "Cleaning up Ray on all nodes..."
ray stop --force 2>/dev/null || true
for N in $(seq 1 "$NUM_WORKERS"); do
    NODE_VAR="WORKER${N}_NODE"
    JOB_VAR="WORKER${N}_JOB"
    NODE=${!NODE_VAR}
    JOB=${!JOB_VAR}
    ssh_remote "$NODE" "$JOB" "ray stop --force 2>/dev/null || true" &
done
wait
echo "Done."
