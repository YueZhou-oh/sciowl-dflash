#!/usr/bin/env bash
# Self-contained runtime env for this pack. Does not source the parent repo.
# Conda and CANN stay installed on the machine; this file only points at them.
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_source_file() {
    local _nounset=0
    [[ $- == *u* ]] && _nounset=1 && set +u
    # shellcheck disable=SC1090
    source "$@"
    local _rc=$?
    [[ "${_nounset}" -eq 1 ]] && set -u
    return "${_rc}"
}

: "${HEMI_ROOT:=/llm_infer_workspace/hedi}"
: "${CONDA_ENV:=${HEMI_ROOT}/conda_unpack/dflash}"
: "${CANN9:=${HEMI_ROOT}/env_own/Ascend_9.0.1/cann/cann-9.0.1}"
: "${CANN85_ROOT:=${HEMI_ROOT}/env_own/Ascend_8.5.1_153}"

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: dflash python not found: ${CONDA_ENV}/bin/python" >&2
    exit 1
fi
if [[ ! -d "${CANN9}" ]]; then
    echo "ERROR: CANN 9.0.1 not found: ${CANN9}" >&2
    exit 1
fi

# conda libstdc++ must stay ahead of CANN's older libstdc++.
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}"

export ASCEND_HOME_PATH="${CANN9}"
export ASCEND_AICPU_PATH="${CANN9}"
export ASCEND_TOOLKIT_HOME="${CANN9}"
export ASCEND_OPP_PATH="${CANN9}/opp"
export TOOLCHAIN_HOME="${CANN9}/toolkit"
export CANN_HOME="${CANN9}"
ARCH="$(uname -m)"
export LD_LIBRARY_PATH="${CANN9}/lib64:${CANN9}/aarch64-linux/lib64:${CANN9}/lib64/plugin/opskernel:${CANN9}/lib64/plugin/nnengine:${CANN9}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}:${CANN9}/tools/aml/lib64:${CANN9}/tools/aml/lib64/plugin:${LD_LIBRARY_PATH}"
export PYTHONPATH="${CANN9}/python/site-packages:${CANN9}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"
export PATH="${CONDA_ENV}/bin:${CANN9}/bin:${CANN9}/tools/hcc/bin:${CANN9}/tools/ccec_compiler/bin:${PATH}"

# ATB ships with the 8.5.1 tree on this machine. Load it when present.
: "${ZSH_VERSION:=}"
if [[ -f "${CANN85_ROOT}/nnal/atb/set_env.sh" ]]; then
    _source_file "${CANN85_ROOT}/nnal/atb/set_env.sh"
fi

export CC="${PACK_DIR}/scripts/triton_host_cxx.sh"
export CXX="${CC}"
export TRITON_NPU_COMPILER_PATH="${CANN9}/tools/ccec_compiler/bin"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_sciowl_dflash}"
mkdir -p "${TRITON_CACHE_DIR}"

# NPU adapter lives inside this pack. sitecustomize.py is on PYTHONPATH.
export PYTHONPATH="${PACK_DIR}/npu_adapter:${PYTHONPATH}"
export SPECFORGE_NPU_ENABLE=1
export TASK_QUEUE_ENABLE=1
export VLLM_KV_CACHE_LAYOUT=LBHNC
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"
export VLLM_ASCEND_ENABLE_CUSTOM_OP="${VLLM_ASCEND_ENABLE_CUSTOM_OP:-0}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"

stub="${RTMETA_STUB:-${HEMI_ROOT}/env_own/librtmeta_stub.so}"
if [[ -f "${stub}" ]]; then
    export LD_PRELOAD="${stub}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY VLLM_PORT
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PY="${CONDA_ENV}/bin/python"

export PACK_DIR
export TARGET_MODEL="${TARGET_MODEL:-${HEMI_ROOT}/qwen-llama-eagle/SciOWL-Instruct-30B-A3B}"
export DRAFT_CKPT="${DRAFT_CKPT:-${PACK_DIR}/ckpt/dflash_epoch4}"
export SERVE_PORT="${SERVE_PORT:-18180}"
export SERVE_TP="${SERVE_TP:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
