#!/usr/bin/env bash
# SciOWL target only. Single node, 8 NPUs, tensor parallel 8, context 4096.
# This is the no-DFlash side of the 48-concurrency task sweep.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/serve_env.sh"
if [[ ! -d "${TARGET_MODEL}" ]]; then
    echo "ERROR: target model not found: ${TARGET_MODEL}" >&2
    exit 1
fi
echo "baseline target=${TARGET_MODEL} tp=${SERVE_TP} port=${SERVE_PORT} devices=${ASCEND_RT_VISIBLE_DEVICES}"
cd "${PACK_DIR}"
exec "${PY}" -m vllm.entrypoints.cli.main serve "${TARGET_MODEL}" \
    --host 127.0.0.1 \
    --port "${SERVE_PORT}" \
    --tensor-parallel-size "${SERVE_TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --no-enable-prefix-caching
