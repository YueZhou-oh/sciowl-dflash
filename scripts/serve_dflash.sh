#!/usr/bin/env bash
# SciOWL target plus the packed DFlash draft.
# Uses only programs in this pack plus the machine conda/CANN install.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/serve_env.sh"

if [[ ! -f "${DRAFT_CKPT}/model.safetensors" || ! -f "${DRAFT_CKPT}/config.json" ]]; then
    echo "ERROR: draft checkpoint incomplete: ${DRAFT_CKPT}" >&2
    exit 1
fi
if [[ ! -d "${TARGET_MODEL}" ]]; then
    echo "ERROR: target model not found: ${TARGET_MODEL}" >&2
    exit 1
fi

echo "dflash target=${TARGET_MODEL}"
echo "dflash draft=${DRAFT_CKPT} tp=${SERVE_TP} port=${SERVE_PORT}"
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
    --no-enable-prefix-caching \
    --spec-model "${DRAFT_CKPT}" \
    --spec-method dflash \
    --spec-tokens 15
