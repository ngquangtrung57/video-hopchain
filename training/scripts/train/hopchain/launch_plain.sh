#!/bin/bash
# Plain GRPO arm: exploration off. Set the node names and scheduler job ids in
# the environment before calling this script.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export REPO
cd "$REPO"

: "${HEAD_NODE:?set HEAD_NODE}"
: "${HEAD_JOB:?set HEAD_JOB}"
: "${WORKER1_NODE:?set WORKER1_NODE}"
: "${WORKER1_JOB:?set WORKER1_JOB}"
: "${WORKER2_NODE:?set WORKER2_NODE}"
: "${WORKER2_JOB:?set WORKER2_JOB}"
: "${WORKER3_NODE:?set WORKER3_NODE}"
: "${WORKER3_JOB:?set WORKER3_JOB}"
export HEAD_NODE HEAD_JOB WORKER1_NODE WORKER1_JOB WORKER2_NODE WORKER2_JOB WORKER3_NODE WORKER3_JOB
unset WORKER4_NODE WORKER5_NODE

export EXP_NAME_OVERRIDE=${EXP_NAME_OVERRIDE:-hopchain_grpo}

export EXPLORE_ENABLE=false
export CGE_ENABLE=false

exec bash scripts/train/hopchain/run_4node_hopchain.sh
