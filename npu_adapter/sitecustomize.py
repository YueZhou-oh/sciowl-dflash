"""
sitecustomize.py - Auto-activation for NPU adapter.

Python automatically executes sitecustomize.py at startup when it's
in the Python path (site-packages or PYTHONPATH).

When SPECFORGE_NPU_ENABLE=1, this module automatically applies
CUDA-to-NPU patches before any user code runs.

SpecForge patch: Ray's internal agents (dashboard agent, runtime-env
agent, autoscaler monitor, log monitor) do NOT need torch/NPU, but
importing npu_adapter (which loads torch_npu .so) adds ~18s to their
startup. That exceeds the raylet's 15s agent_register_timeout (not
reliably overridable on this build), so the raylet crashes with
"Timed out waiting for file metrics_agent_port". Skip the heavy import
for those agent scripts; keep it for ray workers (default_worker /
setup_worker, where vLLM actors run) and all user code.
"""
import os
import sys

# Triton cache on the shared filesystem races when many workers compile the
# same kernel (os.replace of a temp file that a sibling already renamed).
# Give each process a node-local cache.
_triton_cache = os.environ.get("TRITON_CACHE_DIR", "")
if _triton_cache.startswith("/llm_infer_workspace") or _triton_cache.startswith("/home"):
    _local_cache = f"/tmp/triton_cache_{os.getpid()}"
    os.makedirs(_local_cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = _local_cache

if os.environ.get("SPECFORGE_NPU_ENABLE", "0") == "1":
    _argv0 = sys.argv[0] if sys.argv else ""
    _skip_npu = (
        _argv0.endswith("ray/dashboard/agent.py")
        or _argv0.endswith("ray/_private/runtime_env/agent/main.py")
        or _argv0.endswith("ray/_private/monitor.py")
        or _argv0.endswith("ray/_private/log_monitor.py")
        or _argv0.endswith("log_monitor.py")
    )
    if not _skip_npu:
        try:
            import npu_adapter

            npu_adapter.register()
        except Exception as e:
            print(f"[SpecForge-NPU] Auto-registration failed: {e}", file=sys.stderr)
