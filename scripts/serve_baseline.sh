#!/usr/bin/env bash
# SciOWL target only. Single node, 8 NPUs, tensor parallel 8, context 4096.
# This is the no-DFlash side of the 48-concurrency task sweep.
source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh
source /dpc-zhouy/usr/local/Ascend/cann/set_env.sh
source /dpc-zhouy/usr/local/Ascend/nnal/atb/set_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PY="/usr/local/python3.12.13/bin/python"

export PACK_DIR="/dpc-zhouy/zhouy/sciowl_dflash_infer_pack"
export TARGET_MODEL="/workspace/SciOWL-Instruct-30B-A3B"
export SERVE_PORT="${SERVE_PORT:-8005}"
export SERVE_TP="${SERVE_TP:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

export TASK_QUEUE_ENABLE=1
export VLLM_LOG_STATS_INTERVAL=10

TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')

# export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"
# export VLLM_ASCEND_ENABLE_CUSTOM_OP="${VLLM_ASCEND_ENABLE_CUSTOM_OP:-0}"
# export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"

# source "${SCRIPT_DIR}/serve_env.sh"
if [[ ! -d "${TARGET_MODEL}" ]]; then
    echo "ERROR: target model not found: ${TARGET_MODEL}" >&2
    exit 1
fi
echo "baseline target=${TARGET_MODEL} tp=${SERVE_TP} port=${SERVE_PORT} devices=${ASCEND_RT_VISIBLE_DEVICES}"
cd "${PACK_DIR}"
exec "${PY}" -m vllm.entrypoints.cli.main serve "${TARGET_MODEL}" \
    --host 0.0.0.0 \
    --port "${SERVE_PORT}" \
    --served-model-name sciowl \
    --tensor-parallel-size "${SERVE_TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --no-enable-prefix-caching >"/dpc-zhouy/zhouy/sciowl_dflash_infer_pack/logs/baseline_${TIMESTAMP}.log" 2>&1 &
