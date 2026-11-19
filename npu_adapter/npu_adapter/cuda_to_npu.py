"""
CUDA-to-NPU transparent adapter for SpecForge/any PyTorch project.

When activated (via SPECFORGE_NPU_ENABLE=1), this module monkey-patches
torch so that CUDA calls are transparently redirected to NPU equivalents,
allowing code written for CUDA GPUs to run on Ascend NPUs without modification.

Mechanism (inspired by npuslim's sitecustomize + plugin registration pattern):
1. Detect NPU availability via torch.npu / ASCEND_HOME_PATH
2. Replace key torch.cuda methods with NPU equivalents
3. Make torch.cuda.is_available() return True when NPU is available
4. Redirect device strings ("cuda" -> "npu", "cuda:N" -> "npu:N")

Usage:
    # Method 1: Environment variable (recommended)
    export SPECFORGE_NPU_ENABLE=1
    python your_script.py

    # Method 2: Use the wrapper script
    bash npu_adapter/run_with_npu.sh your_script.sh

    # Method 3: Explicit import (must be FIRST import)
    import npu_adapter
    npu_adapter.register()
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from typing import Any

import torch

_PATCHED = False
_COMPAT_PATCHED = False
_ORIGINALS: dict[str, Any] = {}


def _is_npu_available() -> bool:
    """Detect whether Ascend NPU is available."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return True
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_home and os.path.isdir(ascend_home):
        return True
    return False


def _save_original(module, attr_name: str) -> None:
    """Save original attribute before patching."""
    key = f"{module.__name__}.{attr_name}" if hasattr(module, "__name__") else attr_name
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(module, attr_name, None)


def _log_patch(target: str, reason: str = "") -> None:
    """Log a patch being applied."""
    msg = f"[SpecForge-NPU] Patched: {target}"
    if reason:
        msg += f" ({reason})"
    print(msg, file=sys.stderr)


def _patch_torch_cuda() -> None:
    """Patch torch.cuda module to redirect to torch.npu."""
    if not hasattr(torch, "cuda"):
        return

    cuda_mod = torch.cuda
    npu_mod = getattr(torch, "npu", None)

    # -- is_available() -> return True when NPU is present --
    _save_original(cuda_mod, "is_available")
    original_is_available = cuda_mod.is_available

    @functools.wraps(original_is_available)
    def patched_is_available() -> bool:
        if _is_npu_available():
            return True
        return original_is_available()

    cuda_mod.is_available = patched_is_available
    _log_patch("torch.cuda.is_available", "returns True for NPU too")

    # -- device_count() --
    _save_original(cuda_mod, "device_count")
    original_device_count = cuda_mod.device_count

    @functools.wraps(original_device_count)
    def patched_device_count() -> int:
        if npu_mod is not None:
            try:
                return npu_mod.device_count()
            except Exception:
                pass
        return original_device_count()

    cuda_mod.device_count = patched_device_count
    _log_patch("torch.cuda.device_count", "delegates to npu when available")

    # -- synchronize() --
    if npu_mod is not None:
        _save_original(cuda_mod, "synchronize")
        original_sync = cuda_mod.synchronize

        @functools.wraps(original_sync)
        def patched_synchronize(device=None):
            if _is_npu_available():
                try:
                    npu_mod.synchronize()
                    return
                except Exception:
                    pass
            original_sync(device)

        cuda_mod.synchronize = patched_synchronize
        _log_patch("torch.cuda.synchronize", "redirects to torch.npu.synchronize")

    # -- manual_seed_all() --
    if npu_mod is not None:
        _save_original(cuda_mod, "manual_seed_all")
        original_manual_seed = cuda_mod.manual_seed_all

        @functools.wraps(original_manual_seed)
        def patched_manual_seed_all(seed: int):
            if _is_npu_available():
                try:
                    npu_mod.manual_seed_all(seed)
                except Exception:
                    pass
            original_manual_seed(seed)

        cuda_mod.manual_seed_all = patched_manual_seed_all
        _log_patch("torch.cuda.manual_seed_all", "also seeds npu")

    # -- get_rng_state() / set_rng_state() --
    # torch._dynamo.utils.preserve_rng_state() calls these when cuda.is_available()
    # is True; without redirecting, inductor init crashes on NPU-only builds.
    if npu_mod is not None:
        for attr in ("get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all"):
            if not hasattr(cuda_mod, attr) or not hasattr(npu_mod, attr):
                continue
            _save_original(cuda_mod, attr)
            original_fn = _ORIGINALS[f"torch.cuda.{attr}"]
            npu_target = getattr(npu_mod, attr)

            @functools.wraps(original_fn)
            def patched_rng(*args, _npu_target=npu_target, _original=original_fn, **kwargs):
                if _is_npu_available():
                    try:
                        return _npu_target(*args, **kwargs)
                    except Exception:
                        pass
                return _original(*args, **kwargs)

            setattr(cuda_mod, attr, patched_rng)

        _log_patch(
            "torch.cuda.get_rng_state/set_rng_state",
            "redirects to torch.npu for inductor/dynamo",
        )

    # -- current_device() --
    if npu_mod is not None:
        _save_original(cuda_mod, "current_device")
        original_current_device = cuda_mod.current_device

        @functools.wraps(original_current_device)
        def patched_current_device() -> int:
            if _is_npu_available():
                try:
                    return npu_mod.current_device()
                except Exception:
                    pass
            return original_current_device()

        cuda_mod.current_device = patched_current_device
        _log_patch("torch.cuda.current_device", "delegates to npu when available")

    # -- set_device() --
    if npu_mod is not None:
        _save_original(cuda_mod, "set_device")
        original_set_device = cuda_mod.set_device

        @functools.wraps(original_set_device)
        def patched_set_device(device):
            if _is_npu_available():
                try:
                    npu_mod.set_device(device)
                    return
                except Exception:
                    pass
            original_set_device(device)

        cuda_mod.set_device = patched_set_device
        _log_patch("torch.cuda.set_device", "redirects to torch.npu.set_device")

    # -- get_device_properties() / get_device_capability() --
    # torch._dynamo / torch.utils._triton probe CUDA capability during compile;
    # is_available() alone is not enough on NPU-only torch builds.
    if npu_mod is not None:

        class _NPUDevicePropertiesProxy:
            """NPU device props with CUDA-compatible major/minor for dynamo/triton."""

            def __init__(self, npu_props: Any) -> None:
                self._npu_props = npu_props

            def __getattr__(self, name: str) -> Any:
                if name == "major":
                    return 8
                if name == "minor":
                    return 0
                if name == "multi_processor_count":
                    count = getattr(self._npu_props, "multi_processor_count", None)
                    if count is not None:
                        return count
                    return getattr(self._npu_props, "cube_core_num", 24)
                return getattr(self._npu_props, name)

            def __repr__(self) -> str:
                return f"_NPUDevicePropertiesProxy({self._npu_props!r})"

        def _resolve_npu_device_index(device: Any) -> int:
            if device is None:
                return npu_mod.current_device()
            if isinstance(device, torch.device):
                return device.index if device.index is not None else 0
            if isinstance(device, str):
                if ":" in device:
                    return int(device.rsplit(":", 1)[-1])
                return 0
            return int(device)

        if hasattr(cuda_mod, "get_device_properties"):
            _save_original(cuda_mod, "get_device_properties")
            original_get_device_properties = cuda_mod.get_device_properties

            @functools.wraps(original_get_device_properties)
            def patched_get_device_properties(device=None):
                if _is_npu_available():
                    idx = _resolve_npu_device_index(device)
                    return _NPUDevicePropertiesProxy(npu_mod.get_device_properties(idx))
                return original_get_device_properties(device)

            cuda_mod.get_device_properties = patched_get_device_properties

        if hasattr(cuda_mod, "get_device_capability"):
            _save_original(cuda_mod, "get_device_capability")
            original_get_device_capability = cuda_mod.get_device_capability

            @functools.wraps(original_get_device_capability)
            def patched_get_device_capability(device=None):
                if _is_npu_available():
                    return (8, 0)
                return original_get_device_capability(device)

            cuda_mod.get_device_capability = patched_get_device_capability

        _log_patch(
            "torch.cuda.get_device_properties/get_device_capability",
            "delegates to npu with fake CUDA capability for dynamo/triton",
        )

    # -- empty_cache() --
    if npu_mod is not None:
        _save_original(cuda_mod, "empty_cache")
        original_empty_cache = cuda_mod.empty_cache

        @functools.wraps(original_empty_cache)
        def patched_empty_cache():
            if _is_npu_available():
                try:
                    npu_mod.empty_cache()
                except Exception:
                    pass
            original_empty_cache()

        cuda_mod.empty_cache = patched_empty_cache
        _log_patch("torch.cuda.empty_cache", "also clears npu cache")

    # -- Stream class -> wrap to use NPU stream --
    if npu_mod is not None and hasattr(cuda_mod, "Stream"):
        _save_original(cuda_mod, "Stream")
        original_stream_class = cuda_mod.Stream
        npu_stream_class = getattr(npu_mod, "Stream", None)

        class NPUCompatibleStream(original_stream_class):
            """Stream wrapper that falls back to NPU stream when needed."""
            def __new__(cls, device=None, *args, **kwargs):
                if _is_npu_available() and npu_stream_class is not None:
                    obj = object.__new__(cls)
                    return obj
                return super().__new__(cls)

            def __init__(self, device=None, *args, **kwargs):
                self._npu_stream = None
                if _is_npu_available() and npu_stream_class is not None:
                    self._npu_stream = npu_stream_class(device=device, *args, **kwargs)
                else:
                    super().__init__(device, *args, **kwargs)

            def wait_stream(self, stream):
                if self._npu_stream is not None:
                    self._npu_stream.wait_stream(stream._npu_stream if hasattr(stream, "_npu_stream") else stream)
                else:
                    super().wait_stream(stream)

            def record_event(self, event=None):
                if self._npu_stream is not None:
                    return self._npu_stream.record_event(event)
                return super().record_event(event)

            def synchronize(self):
                if self._npu_stream is not None:
                    self._npu_stream.synchronize()
                else:
                    super().synchronize()

            def query(self):
                if self._npu_stream is not None:
                    return self._npu_stream.query()
                return super().query()

            def __enter__(self):
                if self._npu_stream is not None:
                    self._npu_stream.__enter__()
                else:
                    super().__enter__()
                return self

            def __exit__(self, *args):
                if self._npu_stream is not None:
                    self._npu_stream.__exit__(*args)

        cuda_mod.Stream = NPUCompatibleStream
        _log_patch("torch.cuda.Stream", "NPU-compatible wrapper")


def _patch_torch_device() -> None:
    """Patch torch.device to translate 'cuda' -> 'npu' when NPU is active.

    torch.device is a C extension type (torch.Device) that cannot be
    subclassed or replaced with a plain function (it breaks ``str|torch.device``
    type annotations).  Instead we store the original and provide a helper
    that callers can use to translate device strings explicitly.
    """
    if not _is_npu_available():
        return

    # Save the original factory for use by the helper.
    _save_original(torch, "device")

    def npu_device(type_str: str = "cpu", index: int | None = None):
        """Like ``torch.device`` but cuda/cuda:N -> npu/npu:N on NPU."""
        if isinstance(type_str, str):
            stripped = type_str.strip().lower()
            if stripped in ("cuda", "gpu"):
                return torch._original_device("npu", index)
            if stripped.startswith("cuda:"):
                return torch._original_device("npu", int(stripped.split(":", 1)[1]))
        return torch._original_device(type_str, index)

    # Export the NPU-aware helper on the torch namespace so scripts can
    # use ``torch.npu_device("cuda:0")`` instead of ``torch.device("cuda:0")``.
    torch.npu_device = npu_device
    _log_patch("torch.npu_device", "provided as explicit cuda->npu device helper")
    _log_patch("torch.device", "NOT patched — use torch.npu_device() for NPU-safe device creation")


def _patch_cuda_amp() -> None:
    """Patch torch.cuda.amp autocast to use npu autocast when available."""
    if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "amp"):
        return
    if not _is_npu_available():
        return

    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None or not hasattr(npu_mod, "amp"):
        return

    _save_original(torch.cuda.amp, "autocast")
    original_autocast = torch.cuda.amp.autocast

    @functools.wraps(original_autocast)
    def patched_autocast(enabled=True, **kwargs):
        if _is_npu_available():
            return npu_mod.amp.autocast(enabled=enabled, **kwargs)
        return original_autocast(enabled=enabled, **kwargs)

    torch.cuda.amp.autocast = patched_autocast
    _log_patch("torch.cuda.amp.autocast", "redirects to torch.npu.amp.autocast")

    # Also patch GradScaler
    if hasattr(torch.cuda.amp, "GradScaler"):
        _save_original(torch.cuda.amp, "GradScaler")
        original_grad_scaler = torch.cuda.amp.GradScaler
        npu_grad_scaler = getattr(npu_mod.amp, "GradScaler", None)

        if npu_grad_scaler is not None:
            class NPUCompatibleGradScaler(original_grad_scaler):
                def __init__(self, *args, **kwargs):
                    if _is_npu_available():
                        npu_grad_scaler.__init__(self, *args, **kwargs)
                    else:
                        super().__init__(*args, **kwargs)

            torch.cuda.amp.GradScaler = NPUCompatibleGradScaler
            _log_patch("torch.cuda.amp.GradScaler", "NPU-compatible wrapper")


def _patch_cudnn_backend() -> None:
    """Keep the real torch.backends.cudnn module on NPU.

    The original module already exposes callable helpers such as
    ``is_acceptable`` that torch dynamo registers during compile. Replacing
    the module with a no-op proxy breaks dynamo (assert callable).
    Setting ``deterministic`` / ``benchmark`` works without CUDA.
    """
    if not hasattr(torch.backends, "cudnn"):
        return
    if not _is_npu_available():
        return

    _log_patch(
        "torch.backends.cudnn",
        "kept original module (dynamo needs is_acceptable callable)",
    )


def _patch_triton_utils() -> None:
    """Disable CUDA TMA probes on NPU-only torch builds."""
    if not _is_npu_available():
        return

    try:
        import torch.utils._triton as triton_utils
    except ImportError:
        return

    if not hasattr(triton_utils, "_device_supports_tma"):
        return

    _save_original(triton_utils, "_device_supports_tma")
    original_device_supports_tma = triton_utils._device_supports_tma

    @functools.cache
    def patched_device_supports_tma() -> bool:
        if _is_npu_available():
            return False
        return original_device_supports_tma()

    triton_utils._device_supports_tma = patched_device_supports_tma
    _log_patch("torch.utils._triton._device_supports_tma", "returns False on NPU")


def _patch_torch_accelerator() -> None:
    """Redirect torch.accelerator memory APIs to torch.npu on Ascend.

    vLLM 0.21+ uses torch.accelerator.memory_stats(), which does not work on
    torch 2.9 + torch_npu unless redirected. vllm_ascend applies the same
    patch in worker-adjacent platform code, but vLLM worker subprocesses may
    hit MemorySnapshot before that import runs.
    """
    if not _is_npu_available():
        return
    if not hasattr(torch, "accelerator") or not hasattr(torch, "npu"):
        return

    npu_mod = torch.npu
    if not all(hasattr(npu_mod, attr) for attr in ("memory_stats", "memory_reserved", "reset_peak_memory_stats", "empty_cache")):
        return

    def _empty_cache() -> None:
        npu_mod.empty_cache()

    _save_original(torch.accelerator, "empty_cache")
    _save_original(torch.accelerator, "memory_stats")
    _save_original(torch.accelerator, "memory_reserved")
    _save_original(torch.accelerator, "reset_peak_memory_stats")

    torch.accelerator.empty_cache = _empty_cache
    torch.accelerator.memory_stats = npu_mod.memory_stats
    torch.accelerator.memory_reserved = npu_mod.memory_reserved
    torch.accelerator.reset_peak_memory_stats = npu_mod.reset_peak_memory_stats
    _log_patch(
        "torch.accelerator.memory_stats",
        "redirects to torch.npu for vLLM MemorySnapshot",
    )


def _patch_datasets_compat() -> None:
    """Patch datasets library to accept 'List' feature type (backward compat).

    HuggingFace datasets >= 3.x deprecated 'List' in favor of 'Sequence'.
    Arrow files generated by newer datasets contain 'List' in schema metadata,
    which older datasets versions can't parse.  Alias it to 'Sequence'.
    """
    try:
        from datasets.features import features as _ds_features
    except ImportError:
        return
    if "List" not in _ds_features._FEATURE_TYPES:
        _ds_features._FEATURE_TYPES["List"] = _ds_features._FEATURE_TYPES["Sequence"]
        _log_patch("datasets.features", "'List' aliased to 'Sequence' for compat")


def _patch_mistral_common_compat() -> None:
    """Backport mistral_common >=1.11 API needed by vLLM 0.21+.

    vLLM 0.21 requires mistral_common>=1.11.2 (NamedToolChoice, ToolChoiceEnum).
    Offline clusters may still have 1.9.x installed.
    """
    try:
        import mistral_common.protocol.instruct.tool_calls as tc
    except ImportError:
        return

    if hasattr(tc, "NamedToolChoice") and hasattr(tc, "ToolChoiceEnum"):
        return

    from typing import Union

    from mistral_common.base import MistralBase

    class FunctionName(MistralBase):
        name: str

    if not hasattr(tc, "ToolChoiceEnum") and hasattr(tc, "ToolChoice"):
        tc.ToolChoiceEnum = tc.ToolChoice

    if not hasattr(tc, "NamedToolChoice"):

        class NamedToolChoice(MistralBase):
            type: tc.ToolTypes = tc.ToolTypes.function
            function: FunctionName

        tc.NamedToolChoice = NamedToolChoice
        tc.FunctionName = FunctionName

    tc.ToolChoice = Union[tc.ToolChoiceEnum, tc.NamedToolChoice]
    _log_patch(
        "mistral_common.protocol.instruct.tool_calls",
        "backported NamedToolChoice/ToolChoiceEnum for vLLM 0.21",
    )


def _speculative_method_supports(annotation: object, method: str) -> bool:
    """Return True if a SpeculativeConfig.method annotation includes ``method``."""
    import typing

    if annotation is None:
        return False

    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        return any(_speculative_method_supports(arg, method) for arg in typing.get_args(annotation))

    if origin is typing.Literal:
        return method in typing.get_args(annotation)

    args = getattr(annotation, "__args__", ())
    if isinstance(args, tuple) and method in args:
        return True

    return False


def _patch_ascend_mla_kv_spec_compat() -> None:
    """AscendMLAAttentionSpec.__hash__ fails on specs pickled before platform patch."""
    try:
        import dataclasses

        import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa: F401
        from vllm_ascend.core.recompute_scheduler import register_ascend_mla_spec_in_manager
        from vllm_ascend.patch.platform.patch_kv_cache_interface import (
            AscendMLAAttentionSpec,
            _get_c8_k_cache_dtype,
            _get_c8_k_scale_cache_dtype,
        )
    except ImportError:
        return

    import torch

    _defaults: dict[str, object] = {
        "scale_dim": 0,
        "scale_dtype": torch.int8,
        "sparse_head_dim": None,
        "cache_sparse_c8": False,
    }
    _field_names = tuple(f.name for f in dataclasses.fields(AscendMLAAttentionSpec))

    def _field_value(self: AscendMLAAttentionSpec, name: str) -> object:
        if hasattr(self, name):
            return getattr(self, name)
        if name == "c8_k_cache_dtype":
            return _get_c8_k_cache_dtype()
        if name == "c8_k_scale_cache_dtype":
            return _get_c8_k_scale_cache_dtype()
        if name in _defaults:
            return _defaults[name]
        raise AttributeError(name)

    def _safe_hash(self: AscendMLAAttentionSpec) -> int:
        return hash(tuple(_field_value(self, name) for name in _field_names))

    AscendMLAAttentionSpec.__hash__ = _safe_hash  # type: ignore[method-assign]

    register_ascend_mla_spec_in_manager()
    _log_patch(
        "AscendMLAAttentionSpec.__hash__",
        "compat for unpickled specs missing c8_k_cache_dtype",
    )


def _patch_vllm_ascend_platform_early() -> None:
    """Apply vllm_ascend platform patches in every subprocess before KV cache init."""
    try:
        from vllm_ascend.utils import adapt_patch
    except ImportError:
        return
    adapt_patch(is_global_patch=True)
    _patch_ascend_mla_kv_spec_compat()
    _log_patch("vllm_ascend.platform", "loaded early in subprocess via npu_adapter")


def _should_disable_vllm_ascend_custom_op() -> bool:
    """CANN 8.5.x lacks aclnnAddRmsNormBias; use torch_npu.npu_add_rms_norm fallback."""
    env = os.environ.get("VLLM_ASCEND_ENABLE_CUSTOM_OP", "").lower()
    if env in {"1", "true", "yes"}:
        return False
    if env in {"0", "false", "no"}:
        return True
    cann = os.environ.get("CANN_HOME") or os.environ.get("ASCEND_HOME_PATH") or ""
    return "cann-8.5" in cann


def _patch_glm_rot_weight() -> None:
    """QuaRot checkpoints ship ``rot.weight`` for MTP, not the target model.

    ``GlmMoeDsaForCausalLM`` has no ``rot`` module, so AutoWeightsLoader
    rejects the tensor. Speculative MTP applies the rotation itself.
    """
    try:
        from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
        from vllm.model_executor.models.utils import AutoWeightsLoader
    except ImportError:
        return
    if getattr(GlmMoeDsaForCausalLM.load_weights, "_specforge_rot", False):
        return

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["rot"])
        return loader.load_weights(weights)

    load_weights._specforge_rot = True
    GlmMoeDsaForCausalLM.load_weights = load_weights


def _compiling() -> bool:
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


_W8A8_SAFE_ROWS = 128


def _patch_quant_matmul_rows() -> None:
    """CANN 8.5.1 W8A8 matmul is correct through 128 rows and corrupts past that.

    Row chunks are independent, so slicing the activation (and the per-token
    scale) matches one full matmul.
    """
    try:
        import torch
        import torch_npu
    except ImportError:
        return
    current = getattr(torch_npu, "npu_quant_matmul", None)
    if current is None or getattr(current, "_specforge_row_chunk", False):
        return

    def _quant_matmul(x1, x2, scale, **kwargs):
        rows = x1.shape[0] if x1.dim() == 2 else 0
        if _compiling() or rows <= _W8A8_SAFE_ROWS:
            return current(x1, x2, scale, **kwargs)
        pertoken = kwargs.get("pertoken_scale")
        parts = []
        for start in range(0, rows, _W8A8_SAFE_ROWS):
            stop = min(start + _W8A8_SAFE_ROWS, rows)
            kw = dict(kwargs)
            if pertoken is not None:
                kw["pertoken_scale"] = pertoken[start:stop]
            parts.append(current(x1[start:stop], x2, scale, **kw))
        return torch.cat(parts, dim=0)

    _quant_matmul._specforge_row_chunk = True  # type: ignore[attr-defined]
    torch_npu.npu_quant_matmul = _quant_matmul
    _log_patch(
        "torch_npu.npu_quant_matmul",
        "W8A8 rows above 128 are split; CANN 8.5.1 corrupts the longer call",
    )


_MOE_RUNNERS: dict[int, Any] = {}


def _chunk_moe_rows(runner: Any, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    # Read the function from the class. An instance lookup would bind self
    # and then the explicit runner argument makes six positional args.
    original = type(runner)._specforge_original_forward
    rows = hidden_states.shape[0]
    if rows <= _W8A8_SAFE_ROWS:
        return original(runner, hidden_states, router_logits, None, None)
    parts = []
    for start in range(0, rows, _W8A8_SAFE_ROWS):
        stop = min(start + _W8A8_SAFE_ROWS, rows)
        logits = router_logits[start:stop] if router_logits.shape[0] == rows else router_logits
        parts.append(original(runner, hidden_states[start:stop], logits, None, None))
    return torch.cat(parts, dim=0)


def _patch_moe_token_chunks() -> None:
    """MoE dispatch corrupts once a step exceeds 128 tokens.

    The chunked call is a custom op, so Dynamo keeps it as a leaf and the
    split still runs for eager prefill. Decode steps are at most 128 tokens
    and take the single-call path that ACL graphs capture.
    """
    try:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    except ImportError:
        return
    if getattr(AscendMoERunner.forward, "_specforge_token_chunk", False):
        return
    original = AscendMoERunner.forward

    if not hasattr(torch.ops, "specforge") or not hasattr(torch.ops.specforge, "chunked_moe"):

        @torch.library.custom_op("specforge::chunked_moe", mutates_args=())
        def chunked_moe(hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_key: int) -> torch.Tensor:
            return _chunk_moe_rows(_MOE_RUNNERS[int(layer_key)], hidden_states, router_logits)

        @chunked_moe.register_fake
        def _chunked_moe_fake(hidden_states, router_logits, layer_key):
            return torch.empty_like(hidden_states)

    def forward(self, hidden_states, router_logits, input_ids=None, shared_experts_input=None):
        if input_ids is not None or shared_experts_input is not None or hidden_states.dim() != 2:
            return original(self, hidden_states, router_logits, input_ids, shared_experts_input)
        key = id(self)
        _MOE_RUNNERS[key] = self
        return torch.ops.specforge.chunked_moe(hidden_states, router_logits, key)

    forward._specforge_token_chunk = True  # type: ignore[attr-defined]
    AscendMoERunner._specforge_original_forward = original  # type: ignore[attr-defined]
    AscendMoERunner.forward = forward
    _log_patch("AscendMoERunner.forward", "MoE steps above 128 tokens are split in a custom op")


def _chunked_bmm_tokens(x1, x2, tokens_on_x1: int):
    """Chunk the token axis of the TransposeBatchMatMul fallback.

    ``tokens_on_x1`` is 1 for (B, N, K) and 0 for (N, B, K). ``x2`` is (N, K, Out).
    Dynamo keeps the single call so decode graphs stay one kernel.
    """
    import torch

    n = x1.shape[tokens_on_x1]
    if _compiling() or n <= _W8A8_SAFE_ROWS:
        if tokens_on_x1 == 1:
            return torch.bmm(x1.transpose(0, 1), x2).transpose(0, 1).contiguous()
        return torch.bmm(x1, x2).transpose(0, 1).contiguous()
    parts = []
    for start in range(0, n, _W8A8_SAFE_ROWS):
        stop = min(start + _W8A8_SAFE_ROWS, n)
        x2_chunk = x2[start:stop]
        if tokens_on_x1 == 1:
            part = torch.bmm(x1[:, start:stop].transpose(0, 1), x2_chunk).transpose(0, 1)
        else:
            part = torch.bmm(x1[start:stop], x2_chunk).transpose(0, 1)
        parts.append(part)
    return torch.cat(parts, dim=1).contiguous()


def _patch_transpose_bmm() -> None:
    """CANN 8.5.1 rejects npu_transpose_batchmatmul for GLM-5.2 MLA shapes.

    The kernel reports ``inner axis < 65536`` and aborts on
    ``b,m,n,k = 2,32,512,192``. The same product is a batched GEMM.
    """
    try:
        import torch
        import torch_npu
    except ImportError:
        return
    current = getattr(torch_npu, "npu_transpose_batchmatmul", None)
    if current is None or getattr(current, "_specforge_bmm", False):
        return

    def _as_perm(value, default):
        if value is None:
            return default
        return tuple(value)

    def _bmm(x1, x2, **kwargs):
        perm_x1 = _as_perm(kwargs.get("perm_x1"), (1, 0, 2))
        perm_x2 = _as_perm(kwargs.get("perm_x2"), (0, 1, 2))
        perm_y = _as_perm(kwargs.get("perm_y"), (1, 0, 2))
        if perm_x2 == (0, 1, 2) and perm_y == (1, 0, 2) and perm_x1 == (1, 0, 2):
            # (B, N, K) x (N, K, Out) -> (B, N, Out)
            return _chunked_bmm_tokens(x1, x2, tokens_on_x1=1)
        if perm_x2 == (0, 1, 2) and perm_y == (1, 0, 2) and perm_x1 == (0, 1, 2):
            # (N, B, K) x (N, K, Out) -> (B, N, Out)
            return _chunked_bmm_tokens(x1, x2, tokens_on_x1=0)
        return current(x1, x2, **kwargs)

    _bmm._specforge_bmm = True  # type: ignore[attr-defined]
    torch_npu.npu_transpose_batchmatmul = _bmm
    _log_patch(
        "torch_npu.npu_transpose_batchmatmul",
        "MLA shapes use torch.bmm; CANN 8.5.1 TransposeBatchMatMul tiling rejects them",
    )


def _patch_vllm_ascend_custom_op() -> None:
    if not _should_disable_vllm_ascend_custom_op():
        return
    try:
        import vllm_ascend.utils as ascend_utils
    except ImportError:
        return

    ascend_utils._CUSTOM_OP_ENABLED = False

    @functools.wraps(ascend_utils.enable_custom_op)
    def _disabled_enable_custom_op() -> bool:
        return False

    ascend_utils.enable_custom_op = _disabled_enable_custom_op  # type: ignore[method-assign]
    _log_patch(
        "vllm_ascend.utils.enable_custom_op",
        "disabled; layernorm uses torch_npu.npu_add_rms_norm on CANN 8.5",
    )


def _should_use_pytorch_topk_topp() -> bool:
    """CANN 8.5.x lacks aclnnApplyTopKTopPCustom in libopapi.so."""
    env = os.environ.get("VLLM_ASCEND_PYTORCH_TOPK_TOPP", "").lower()
    if env in {"0", "false", "no"}:
        return False
    if env in {"1", "true", "yes"}:
        return True
    cann = os.environ.get("CANN_HOME") or os.environ.get("ASCEND_HOME_PATH") or ""
    return "cann-8.5" in cann


def _patch_vllm_ascend_topk_topp() -> None:
    try:
        import vllm_ascend.sample.sampler as sampler_mod
    except ImportError:
        return
    if getattr(sampler_mod, "_specforge_chunked_topk", False):
        return

    # Workers often have CANN 9 on ASCEND_HOME_PATH, so the cann-8.5 check
    # skipped this patch there. The live path is _apply_top_k_top_p_torch_npu,
    # which calls _apply_top_k_top_p_pytorch by global name. Wrap that.
    _orig_topk = sampler_mod._apply_top_k_top_p_pytorch

    def _chunked_topk(logits, k, p, _orig=_orig_topk, chunk=128):
        rows = logits.shape[0]
        if rows <= chunk:
            return _orig(logits, k, p)
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            _orig(
                logits[start:end],
                None if k is None else k[start:end],
                None if p is None else p[start:end],
            )
        return logits

    sampler_mod._apply_top_k_top_p_pytorch = _chunked_topk  # type: ignore[attr-defined]
    sampler_mod._specforge_chunked_topk = True  # type: ignore[attr-defined]
    _log_patch(
        "vllm_ascend.sample.sampler._apply_top_k_top_p_pytorch",
        "top-k/top-p sort chunked to 128 rows (full-batch aclnnSort needs ~4GiB)",
    )


def _patch_grouped_matmul_pad() -> None:
    """Append SPECFORGE_MOE_PAD extra rows to every expert.

    A fixed slot of `pad` rows cannot hold a mixed prefill step (one expert
    can exceed `pad`), and skipping those steps made the pad a no-op on the
    steady-state batch. Adding `pad` rows on top of the real count keeps
    every index in range, needs no host sync, and still widens the cube GEMM.
    Extra rows are dropped before the result is returned.
    """
    pad = int(os.environ.get("SPECFORGE_MOE_PAD", "0") or "0")
    if pad <= 1:
        return
    try:
        import torch
        import torch_npu
    except ImportError:
        return
    if getattr(torch_npu.npu_grouped_matmul, "_specforge_moe_pad", False):
        return
    orig = torch_npu.npu_grouped_matmul

    def _counts_of(group_list, group_list_type, rows):
        if group_list_type == 1:
            return group_list
        if group_list_type == 0:
            zero = group_list.new_zeros(1)
            return torch.diff(group_list, prepend=zero)
        return None

    def _device_pad(hidden, counts):
        # Extra rows sit after each expert's real tokens. Empty experts stay
        # empty in `counts` (flat cumsum); searchsorted(right=True) still
        # assigns real tokens to the next non-empty expert. No host sync.
        experts = counts.shape[0]
        rows = hidden.shape[0]
        padded_counts = counts + pad
        cumsum = torch.cumsum(counts, 0)
        padded_cumsum = torch.cumsum(padded_counts, 0)
        token_idx = torch.arange(rows, device=hidden.device)
        expert = torch.searchsorted(cumsum, token_idx, right=True).clamp(max=experts - 1)
        local = token_idx - (cumsum - counts)[expert]
        dest = (padded_cumsum - padded_counts)[expert] + local
        cap = rows + experts * pad
        buf = hidden.new_empty((cap, hidden.shape[1]))
        buf.index_copy_(0, dest, hidden)
        return buf, padded_counts, dest

    def _wrapped(*args, **kwargs):
        if args:
            return orig(*args, **kwargs)
        x = kwargs.get("x")
        group_list = kwargs.get("group_list")
        if group_list is None or not isinstance(x, (list, tuple)) or len(x) != 1:
            return orig(*args, **kwargs)
        hidden = x[0]
        if not torch.is_tensor(hidden) or hidden.dim() != 2 or hidden.shape[0] == 0:
            return orig(*args, **kwargs)
        bias = kwargs.get("bias")
        if bias is not None and not (isinstance(bias, (list, tuple)) and bias[0] is None):
            return orig(*args, **kwargs)
        counts = _counts_of(group_list, int(kwargs.get("group_list_type", 0)), hidden.shape[0])
        if counts is None or counts.dim() != 1 or counts.shape[0] == 0:
            return orig(*args, **kwargs)
        gathered, padded_counts, dest = _device_pad(hidden, counts)
        kwargs["x"] = [gathered]
        kwargs["group_list"] = padded_counts
        kwargs["group_list_type"] = 1
        out = orig(*args, **kwargs)

        def _unpad(tensor):
            if not torch.is_tensor(tensor) or tensor.shape[0] != gathered.shape[0]:
                return tensor
            return tensor.index_select(0, dest)

        if isinstance(out, (list, tuple)):
            return type(out)(_unpad(t) for t in out)
        return _unpad(out)

    _wrapped._specforge_moe_pad = True  # type: ignore[attr-defined]
    torch_npu.npu_grouped_matmul = _wrapped
    _log_patch(
        "torch_npu.npu_grouped_matmul",
        f"append {pad} extra rows on every expert inside grouped matmul, then gather back",
    )


def _patch_kv_int8() -> None:
    """Serve bf16 weights with a static int8 KV cache.

    Device sampling showed the vector cores memory-bound about 3x (HBM load
    vs vector math). Decode attention rereads the whole cache each step, so
    halving those bytes is the lever that can raise cube occupancy. Scales are
    a fixed 1/16 per channel; this checkpoint has no calibrated C8 scales.
    Set SPECFORGE_KV_INT8=1 and pass --kv-cache-dtype int8.
    """
    if os.environ.get("SPECFORGE_KV_INT8", "0") != "1":
        return
    try:
        import torch
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionBackendImpl,
            AscendC8AttentionBackendImpl,
        )
    except ImportError:
        return
    if getattr(AscendAttentionBackendImpl.__init__, "_specforge_kv_int8", False):
        return
    orig_init = AscendAttentionBackendImpl.__init__
    orig_prepare = AscendC8AttentionBackendImpl._prepare_c8_scales

    def _init(self, *args, **kwargs):
        # The stock ctor rejects int8 unless a modelslim quant config is
        # present. Keep construction on bf16, then switch the instance onto
        # the C8 forward, which quantizes on the way into the cache.
        if len(args) >= 7:
            args = list(args)
            args[6] = "bfloat16"
            args = tuple(args)
        if "kv_cache_dtype" in kwargs:
            kwargs["kv_cache_dtype"] = "bfloat16"
        orig_init(self, *args, **kwargs)
        self.enable_c8_quant = True
        self.kv_cache_dtype = torch.int8
        self.__class__ = AscendC8AttentionBackendImpl

    def _prepare(self, layer, device):
        if not hasattr(layer, "k_cache_scale"):
            n = self.num_kv_heads * self.head_size
            scale = torch.full((n,), 1.0 / 16.0, dtype=torch.float32, device=device)
            zero = torch.zeros(n, dtype=torch.float32, device=device)
            layer.k_cache_scale = torch.nn.Parameter(scale, requires_grad=False)
            layer.k_cache_offset = torch.nn.Parameter(zero.clone(), requires_grad=False)
            layer.v_cache_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
            layer.v_cache_offset = torch.nn.Parameter(zero.clone(), requires_grad=False)
        return orig_prepare(self, layer, device)

    _init._specforge_kv_int8 = True  # type: ignore[attr-defined]
    AscendAttentionBackendImpl.__init__ = _init  # type: ignore[method-assign]
    AscendC8AttentionBackendImpl._prepare_c8_scales = _prepare  # type: ignore[method-assign]
    _log_patch(
        "AscendAttentionBackendImpl",
        "int8 KV cache via C8 attention (static scale 1/16, no quant checkpoint)",
    )


def _patch_force_paged_attn() -> None:
    """Use paged attention for every decode step, including eager mode.

    Stock vllm_ascend only selects it when cudagraph_mode is FULL_DECODE_ONLY
    and the token count is in pa_shape_list. Capturing that graph segfaults
    in ATB on this CANN, so SPECFORGE_FORCE_PA=1 selects the eager kernel
    instead and never records it into an ACL graph.
    """
    if os.environ.get("SPECFORGE_FORCE_PA", "0") != "1":
        return
    try:
        from vllm_ascend.attention import utils as attn_utils
    except ImportError:
        return
    if getattr(attn_utils.using_paged_attention, "_specforge_force_pa", False):
        return

    def _using(runtime_shape, vllm_config, head_size=None):
        del runtime_shape, vllm_config, head_size
        return True

    _using._specforge_force_pa = True  # type: ignore[attr-defined]
    attn_utils.using_paged_attention = _using
    import sys

    attention_v1 = sys.modules.get("vllm_ascend.attention.attention_v1")
    if attention_v1 is not None:
        attention_v1.using_paged_attention = _using
    _log_patch(
        "vllm_ascend.attention.utils.using_paged_attention",
        "eager paged attention for every decode step (no ACL graph capture)",
    )


def _patch_vllm_compat() -> None:
    """Accept 'extract_hidden_states' on older vLLM versions (<0.21).

    vLLM 0.21+ ships native extract_hidden_states support; skip patching then.
    """
    try:
        from vllm.config.speculative import SpeculativeConfig as _SC
    except ImportError:
        return

    method_field = _SC.__pydantic_fields__.get("method")
    if method_field is not None and _speculative_method_supports(
        method_field.annotation, "extract_hidden_states"
    ):
        return

    try:
        from vllm.engine.arg_utils import EngineArgs as _EA
        from vllm.model_executor.models.registry import ModelRegistry
    except ImportError:
        return

    # Register EAGLE3 variants for Qwen that vLLM 0.15.0 doesn't have natively.
    _eagle3_qwen_classes = {
        "Eagle3Qwen3ForCausalLM": "vllm.model_executor.models.llama_eagle3:Eagle3LlamaForCausalLM",
        "Eagle3Qwen3vlForCausalLM": "vllm.model_executor.models.llama_eagle3:Eagle3LlamaForCausalLM",
    }
    for arch, cls_ref in _eagle3_qwen_classes.items():
        if arch not in ModelRegistry.models:
            ModelRegistry.register_model(arch, cls_ref)
            _log_patch("vllm.model_executor.models.registry", f"registered {arch}")

    _original_create = _EA.create_speculative_config

    def _patched_create_spec_config(self, *args: object, **kwargs: object) -> object:
        sc = getattr(self, "speculative_config", None)
        if isinstance(sc, dict) and sc.get("method") == "extract_hidden_states":
            sc["method"] = "eagle3"
            if "model" not in sc and sc.get("num_speculative_tokens") is not None:
                sc["model"] = self.model
        return _original_create(self, *args, **kwargs)

    _EA.create_speculative_config = _patched_create_spec_config  # type: ignore[method-assign]
    _log_patch(
        "vllm.engine.arg_utils",
        "'extract_hidden_states' mapped to 'eagle3' in EngineArgs",
    )


def _patch_cuda_visible_devices() -> None:
    """
    Handle CUDA_VISIBLE_DEVICES -> ASCEND_RT_VISIBLE_DEVICES mapping.

    If CUDA_VISIBLE_DEVICES is set but ASCEND_RT_VISIBLE_DEVICES is not,
    copy the value over so NPU device selection matches what the user intended.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ascend_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")

    if cuda_visible and not ascend_visible:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = cuda_visible
        _log_patch(
            "CUDA_VISIBLE_DEVICES",
            f"copied '{cuda_visible}' to ASCEND_RT_VISIBLE_DEVICES",
        )


_MOE_OVERLAP_SLOT: contextvars.ContextVar | None = None


def _patch_moe_comm_overlap() -> None:
    """Run the shared-expert GEMM while AllToAll waits on a side stream.

    TokenDispatcherWithAll2AllV issues async all-to-all and then waits on the
    CPU before any independent compute is queued, so the shared expert (which
    does not need the dispatched tokens) used to run only after the collective.
    Set SPECFORGE_MOE_COMM_OVERLAP=1 to enqueue that GEMM on the compute stream
    while the collective runs on the comm stream.
    """
    if os.environ.get("SPECFORGE_MOE_COMM_OVERLAP", "0") != "1":
        return

    import contextvars

    import torch
    import torch_npu  # noqa: F401
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all
    from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV
    from vllm_ascend.utils import shared_expert_dp_enabled

    global _MOE_OVERLAP_SLOT
    if _MOE_OVERLAP_SLOT is None:
        _MOE_OVERLAP_SLOT = contextvars.ContextVar("specforge_moe_overlap", default=None)

    if getattr(TokenDispatcherWithAll2AllV, "_specforge_overlap", False):
        return

    def _fire() -> None:
        slot = _MOE_OVERLAP_SLOT.get()
        if not slot or slot.get("out") is not None:
            return
        fn = slot.get("fn")
        if fn is None:
            return
        slot["out"] = fn()

    def token_dispatch(self, token_dispatch_input):
        with_quant = token_dispatch_input.quant.is_int_quant
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        ) = self._dispatch_preprocess(token_dispatch_input.hidden_states, topk_ids)

        dynamic_scale_after_all2all = None
        if with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens
            )
            _, dynamic_scale_after_all2all, scale_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group
            )
            scale_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        ready = torch.npu.current_stream().record_event()
        _, global_input_tokens, token_handle = async_all_to_all(
            permutated_local_input_tokens,
            output_splits,
            input_splits,
            self.ep_group,
            event=ready,
        )
        try:
            _fire()
        finally:
            token_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = (
            self._dispatch_postprocess(
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
                with_quant,
            )
        )
        from vllm_ascend.ops.fused_moe.moe_runtime_args import (
            MoEAllToAllCombineMetadata,
            MoETokenDispatchOutput,
        )

        return MoETokenDispatchOutput(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            combine_metadata=MoEAllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                topk_weights=topk_weights,
                reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
                reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
                hidden_shape=hidden_shape,
                hidden_shape_before_permute=hidden_shape_before_permute,
            ),
        )

    orig_shared = AscendFusedMoE.shared_forward_impl
    orig_forward_shared = AscendFusedMoE._forward_shared_experts

    def shared_forward_impl(self, hidden_states, router_logits):
        slot = {"fn": None, "out": None}

        def _compute():
            if self._shared_experts is None:
                return None
            out = self._shared_experts(hidden_states)
            return out[0] if isinstance(out, tuple) else out

        slot["fn"] = _compute
        token = _MOE_OVERLAP_SLOT.set(slot)
        try:
            return orig_shared(self, hidden_states, router_logits)
        finally:
            _MOE_OVERLAP_SLOT.reset(token)

    def _forward_shared_experts(self, hidden_states, fused_moe_evts):
        slot = _MOE_OVERLAP_SLOT.get()
        if not slot or slot.get("out") is None:
            return orig_forward_shared(self, hidden_states, fused_moe_evts)
        shared_out = slot["out"]
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2} and not shared_expert_dp_enabled():
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out

    TokenDispatcherWithAll2AllV.token_dispatch = token_dispatch
    TokenDispatcherWithAll2AllV._specforge_overlap = True
    AscendFusedMoE.shared_forward_impl = shared_forward_impl
    AscendFusedMoE._forward_shared_experts = _forward_shared_experts
    print("[SpecForge-NPU] MoE AllToAll overlaps shared-expert GEMM", file=sys.stderr)


def _patch_bnsd_reshape_and_cache() -> None:
    """CANN 8.5.1 ScatterPaKvCache has no scatter_mode=NHSD; use scatter_nd.

    Upstream vllm_ascend LBHNC/HND path calls npu_scatter_pa_kv_cache(..., NHSD)
    which tiling-fails. Prefer npu_scatter_nd_update_ with 3D indices into
    [B,H,S,D] (AICore). Always override so site-packages index_put_ is replaced.
    """
    try:
        from vllm_ascend.device.device_op import DeviceOperator
    except ImportError:
        return

    import torch
    import torch_npu

    def _bnsd_scatter_nd(key, value, key_cache, value_cache, slot_mapping):
        _num_blocks, num_heads, block_size, head_size = key_cache.shape
        slots = slot_mapping.reshape(-1).to(torch.int64)
        key = key.contiguous()
        value = value.contiguous()
        # Callers already slice slot_mapping to num_actual_tokens, so the
        # rows are valid. A host .item() here syncs every layer and aborts
        # ACL graph capture ("synchronize captured-stream").
        if slots.numel() == 0:
            return
        blocks = torch.div(slots, block_size, rounding_mode="floor")
        offsets = torch.remainder(slots, block_size)
        num_tokens = key.shape[0]
        head_idx = torch.arange(num_heads, device=key.device, dtype=torch.int64)
        b = blocks.unsqueeze(1).expand(-1, num_heads).reshape(-1)
        h = head_idx.unsqueeze(0).expand(num_tokens, -1).reshape(-1)
        o = offsets.unsqueeze(1).expand(-1, num_heads).reshape(-1)
        updates_k = key.reshape(-1, head_size)
        updates_v = value.reshape(-1, head_size)
        indices = torch.stack(
            (b.to(torch.int32), h.to(torch.int32), o.to(torch.int32)), dim=1
        )
        torch_npu.npu_scatter_nd_update_(key_cache, indices, updates_k)
        torch_npu.npu_scatter_nd_update_(value_cache, indices, updates_v)

    # Prefer mutating the static helper so reshape_and_cache keeps calling it.
    if hasattr(DeviceOperator, "_reshape_and_cache_bnsd_index"):
        DeviceOperator._reshape_and_cache_bnsd_index = staticmethod(_bnsd_scatter_nd)
        _log_patch(
            "DeviceOperator._reshape_and_cache_bnsd_index",
            "LBHNC uses npu_scatter_nd_update_ 3D (AICore; no NHSD on CANN 8.5.1)",
        )
        return

    _orig = DeviceOperator.reshape_and_cache

    @classmethod
    def _patched(cls, key, value, key_cache, value_cache, slot_mapping, use_bnsd=False):
        if use_bnsd and os.environ.get("VLLM_ASCEND_BNSD_SCATTER_NHSD", "0") != "1":
            _bnsd_scatter_nd(key, value, key_cache, value_cache, slot_mapping)
            return
        return _orig(key, value, key_cache, value_cache, slot_mapping, use_bnsd=use_bnsd)

    DeviceOperator.reshape_and_cache = _patched  # type: ignore[method-assign]
    _log_patch(
        "DeviceOperator.reshape_and_cache",
        "LBHNC npu_scatter_nd_update_ 3D (no NHSD on CANN 8.5.1)",
    )


def _apply_compat_patches() -> None:
    """Apply vLLM/datasets/mistral compatibility patches (no NPU required)."""
    global _COMPAT_PATCHED
    if _COMPAT_PATCHED:
        return

    for patch_fn, label in (
        (_patch_moe_comm_overlap, "moe_comm_overlap"),
        (_patch_vllm_ascend_platform_early, "vllm_ascend.platform"),
        (_patch_bnsd_reshape_and_cache, "bnsd_reshape_and_cache"),
        (_patch_datasets_compat, "datasets"),
        (_patch_mistral_common_compat, "mistral_common"),
        (_patch_glm_rot_weight, "glm_rot_weight"),
        (_patch_quant_matmul_rows, "quant_matmul_rows"),
        (_patch_transpose_bmm, "transpose_bmm"),
        (_patch_vllm_ascend_custom_op, "vllm_ascend"),
        (_patch_vllm_ascend_topk_topp, "vllm_ascend.topk_topp"),
        (_patch_grouped_matmul_pad, "grouped_matmul_pad"),
        (_patch_kv_int8, "kv_int8"),
        (_patch_force_paged_attn, "force_paged_attn"),
        (_patch_vllm_compat, "vllm"),
    ):
        try:
            patch_fn()
        except Exception as e:
            warnings.warn(f"Failed to patch {label} compatibility: {e}")

    _COMPAT_PATCHED = True


def register() -> None:
    """
    Apply all CUDA-to-NPU patches.

    Idempotent - safe to call multiple times.
    """
    global _PATCHED

    _apply_compat_patches()

    if _PATCHED:
        return

    if not _is_npu_available():
        print(
            "[SpecForge-NPU] NPU not detected, CUDA->NPU patches skipped.",
            file=sys.stderr,
        )
        return

    print("[SpecForge-NPU] Detected Ascend NPU, applying CUDA->NPU patches...", file=sys.stderr)

    try:
        _patch_cuda_visible_devices()
    except Exception as e:
        warnings.warn(f"Failed to patch CUDA_VISIBLE_DEVICES: {e}")

    try:
        _patch_torch_cuda()
    except Exception as e:
        warnings.warn(f"Failed to patch torch.cuda: {e}")

    try:
        _patch_torch_device()
    except Exception as e:
        warnings.warn(f"Failed to patch torch.device: {e}")

    try:
        _patch_cuda_amp()
    except Exception as e:
        warnings.warn(f"Failed to patch torch.cuda.amp: {e}")

    try:
        _patch_cudnn_backend()
    except Exception as e:
        warnings.warn(f"Failed to patch torch.backends.cudnn: {e}")

    try:
        _patch_torch_accelerator()
    except Exception as e:
        warnings.warn(f"Failed to patch torch.accelerator: {e}")

    try:
        _patch_triton_utils()
    except Exception as e:
        warnings.warn(f"Failed to patch torch.utils._triton: {e}")

    _PATCHED = True
    print("[SpecForge-NPU] All CUDA->NPU patches applied successfully.", file=sys.stderr)


def unregister() -> None:
    """Restore original torch behavior (for testing/debugging)."""
    global _PATCHED
    if not _PATCHED:
        return

    for key, original in _ORIGINALS.items():
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            module_name, attr_name = parts
        else:
            continue

        if module_name == "torch":
            setattr(torch, attr_name, original)
        elif module_name == "torch.cuda":
            setattr(torch.cuda, attr_name, original)
        elif module_name == "torch.cuda.amp":
            setattr(torch.cuda.amp, attr_name, original)
        elif module_name == "torch.backends" and attr_name == "cudnn":
            torch.backends.cudnn = original

    _ORIGINALS.clear()
    _PATCHED = False
    print("[SpecForge-NPU] All patches removed.", file=sys.stderr)
