"""Low-intrusion DeepEP mode and collective probe.

Enable with ``SGLANG_DEEPEP_MODE_PROBE=1``. Optional controls:

* ``SGLANG_DEEPEP_MODE_PROBE_RANKS=0,1``
* ``SGLANG_DEEPEP_MODE_PROBE_MAX_EVENTS=20000``

The probe writes synchronously to stderr so the final BEGIN record survives a
collective hang. It only reads tensor metadata and never synchronizes the NPU.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
import time
from typing import Any, Optional

import torch

_ENABLED = os.getenv("SGLANG_DEEPEP_MODE_PROBE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_MAX_EVENTS = int(os.getenv("SGLANG_DEEPEP_MODE_PROBE_MAX_EVENTS", "20000"))
_RANK_FILTER_RAW = os.getenv("SGLANG_DEEPEP_MODE_PROBE_RANKS", "").strip()
_RANK_FILTER = (
    {int(value) for value in _RANK_FILTER_RAW.split(",") if value.strip()}
    if _RANK_FILTER_RAW
    else None
)

_EVENT_ID = 0
_INSTALLED = False
_LAST_RESOLVED_MODE: Optional[str] = None


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "-1")))


def _parallel_ranks() -> str:
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        fields = []
        for label, name in (
            ("tp", "tp_rank"),
            ("dp", "attn_dp_rank"),
            ("ep", "moe_ep_rank"),
        ):
            value = getattr(parallel, name, None)
            if value is not None:
                fields.append(f"{label}={value}")
        return ",".join(fields) if fields else "unavailable"
    except Exception:
        return "unavailable"


def _forward_mode_from_stack() -> str:
    """Find the nearest ForwardBatch without changing production interfaces."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        for _ in range(48):
            if frame is None:
                break
            forward_batch = frame.f_locals.get("forward_batch")
            if forward_batch is not None:
                mode = getattr(forward_batch, "forward_mode", None)
                if mode is not None:
                    return getattr(mode, "name", str(mode))
            frame = frame.f_back
    finally:
        # A live frame retains the complete model call stack.
        del frame
    return "unknown"


def _mode_state(dispatcher) -> tuple[str, bool, str, str]:
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch

    configured = getattr(dispatcher, "deepep_mode", None)
    configured_name = getattr(configured, "value", str(configured))
    is_extend = bool(get_is_extend_in_batch())
    resolved = configured.resolve(is_extend) if configured is not None else None
    resolved_name = getattr(resolved, "value", str(resolved))
    impl = (
        "_normal_dispatcher"
        if resolved_name == "normal"
        else "_low_latency_dispatcher"
        if resolved_name == "low_latency"
        else "unknown"
    )
    return configured_name, is_extend, resolved_name, impl


def _describe(value: Any, depth: int = 0) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)},dtype={value.dtype},"
            f"device={value.device})"
        )
    if depth >= 1:
        return type(value).__name__
    if isinstance(value, (tuple, list)):
        inner = ",".join(_describe(item, depth + 1) for item in value[:4])
        if len(value) > 4:
            inner += ",..."
        return f"{type(value).__name__}({inner})"
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    fields = []
    for name in ("topk_ids", "topk_weights", "hidden_states"):
        item = getattr(value, name, None)
        if item is not None:
            fields.append(f"{name}={_describe(item, depth + 1)}")
    return f"{type(value).__name__}({','.join(fields)})"


def _emit(dispatcher, method: str, boundary: str, args=(), error=None) -> None:
    global _EVENT_ID, _LAST_RESOLVED_MODE

    rank = _rank()
    if _RANK_FILTER is not None and rank not in _RANK_FILTER:
        return
    if _EVENT_ID >= _MAX_EVENTS:
        return

    _EVENT_ID += 1
    configured, is_extend, resolved, impl = _mode_state(dispatcher)
    switch = (
        f"{_LAST_RESOLVED_MODE}->{resolved}"
        if _LAST_RESOLVED_MODE is not None and _LAST_RESOLVED_MODE != resolved
        else "-"
    )
    _LAST_RESOLVED_MODE = resolved

    stage = getattr(getattr(dispatcher, "_stage", None), "name", "unknown")
    arg_text = ";".join(_describe(arg) for arg in args[:3])
    error_text = "" if error is None else f" error={type(error).__name__}:{error}"
    print(
        "[DeepEP mode probe] "
        f"ts={time.time():.6f} event={_EVENT_ID} pid={os.getpid()} "
        f"rank={rank} parallel={_parallel_ranks()} "
        f"dispatcher=0x{id(dispatcher):x} method={method} boundary={boundary} "
        f"forward_mode={_forward_mode_from_stack()} stage={stage} "
        f"configured={configured} is_extend_in_batch={is_extend} "
        f"resolved={resolved} impl={impl} switch={switch} args=[{arg_text}]"
        f"{error_text}",
        file=sys.stderr,
        flush=True,
    )


def _wrap_method(dispatcher_cls, method_name: str) -> None:
    original = getattr(dispatcher_cls, method_name)
    if getattr(original, "_sglang_deepep_mode_probe", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        _emit(self, method_name, "BEGIN", args)
        try:
            result = original(self, *args, **kwargs)
        except BaseException as error:
            _emit(self, method_name, "ERROR", args, error)
            raise
        _emit(self, method_name, "END", args)
        return result

    wrapped._sglang_deepep_mode_probe = True
    setattr(dispatcher_cls, method_name, wrapped)


def install_deepep_mode_probe(dispatcher_cls) -> bool:
    """Patch DeepEPDispatcher once. Return whether installation occurred."""
    global _INSTALLED
    if not _ENABLED or _INSTALLED:
        return False
    for method_name in (
        "_get_impl",
        "dispatch",
        "dispatch_a",
        "dispatch_b",
        "combine",
        "combine_a",
        "combine_b",
    ):
        _wrap_method(dispatcher_cls, method_name)
    _INSTALLED = True
    print(
        "[DeepEP mode probe] installed "
        f"pid={os.getpid()} ranks={_RANK_FILTER_RAW or 'all'} "
        f"max_events={_MAX_EVENTS}",
        file=sys.stderr,
        flush=True,
    )
    return True

