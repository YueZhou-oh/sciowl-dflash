#!/usr/bin/env bash
# Replay the 8-task, 48-concurrency sweep against a server that is already up.
# Usage: bash run_tasks.sh /path/to/output_dir
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/serve_env.sh"
OUT="${1:-${PACK_DIR}/task_results}"
mkdir -p "${OUT}"
URL="http://127.0.0.1:${SERVE_PORT}"
for task in coding math qa summarization translation writing tool_call science; do
    echo "===== TASK ${task} $(date +%H:%M:%S) ====="
    env -u all_proxy -u ALL_PROXY -u http_proxy -u HTTP_PROXY \
        -u https_proxy -u HTTPS_PROXY \
        NO_PROXY='*' no_proxy='*' \
        python3 -I -S "${SCRIPT_DIR}/bench_tasks.py" \
            --url "${URL}" \
            --model "${TARGET_MODEL}" \
            --task "${task}" \
            --concurrency 48 \
            --requests 48 \
            --max-tokens 3584 \
            --timeout 1800 \
            --out "${OUT}/${task}.json"
done
echo "results: ${OUT}"
