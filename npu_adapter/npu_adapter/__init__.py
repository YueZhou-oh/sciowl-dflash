"""
NPU Adapter for SpecForge - Transparent CUDA-to-NPU migration.

Import and call register() to apply patches, or set SPECFORGE_NPU_ENABLE=1
to activate automatically via sitecustomize.py.

Usage:
    # Option 1: Programmatic
    import npu_adapter
    npu_adapter.register()

    # Option 2: Environment variable (auto-activation)
    export SPECFORGE_NPU_ENABLE=1
    python your_script.py

    # Option 3: Wrapper script
    bash npu_adapter/run_with_npu.sh hed_scripts_offical/your_script.sh
"""

from npu_adapter.cuda_to_npu import register, unregister, _is_npu_available

__all__ = ["register", "unregister", "_is_npu_available"]
