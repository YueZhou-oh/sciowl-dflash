#!/bin/bash
# Host C++ wrapper for Triton NPU: ccec (clang 15) + hcc libstdc++ + clang builtins.
# CANN_HOME must be exported before use. serve_env.sh does that.
set -euo pipefail

if [[ -z "${CANN_HOME:-}" ]]; then
  echo "ERROR: CANN_HOME is not set. Source scripts/serve_env.sh first." >&2
  exit 1
fi

CCEC="${CANN_HOME}/tools/ccec_compiler/bin/ccec"
HCC="${CANN_HOME}/tools/hcc"
TRIP="${HCC}/aarch64-target-linux-gnu"
GCC_VER="$(ls "${HCC}/lib64/gcc/aarch64-target-linux-gnu" 2>/dev/null | head -1)"
CLANG_RES="$("${CCEC}" -print-resource-dir 2>/dev/null)"

if [[ ! -x "${CCEC}" || -z "${GCC_VER}" || -z "${CLANG_RES}" ]]; then
  echo "ERROR: failed to locate ccec/libstdc++ headers under CANN_HOME=${CANN_HOME}" >&2
  exit 1
fi

exec "${CCEC}" \
  -isystem "${CLANG_RES}/include" \
  -isystem "${TRIP}/include/c++/${GCC_VER}" \
  -isystem "${TRIP}/include/c++/${GCC_VER}/aarch64-target-linux-gnu" \
  -isystem "${TRIP}/include" \
  "$@"
