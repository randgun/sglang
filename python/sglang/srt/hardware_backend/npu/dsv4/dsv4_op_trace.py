"""Device-fault attribution probe for NPU DSv4.x / DSpark bring-up.

Why this exists: an AI Core fault is asynchronous. The traceback we get points at
whichever later op happened to sync the stream (for example a tiny H2D copy or an
innocent ``npu.cache_loc_update``), not at the kernel that actually faulted.

How it fixes that: every marker synchronizes the stream *before* printing. Any
pending asynchronous fault therefore raises at the *next* marker, so the last
line printed identifies the launch that broke. Markers also dump the tensor
contract (shape / dtype / stride / contiguity / data_ptr / range) of their
operands, which is what tells an out-of-range or mis-strided argument apart from
a broken op binary.

Enable with ``SGLANG_DSV4_OP_TRACE=1``. It is a no-op otherwise, and safe to
import on any backend, so it can live inside a debug branch.

Reading rule: a tag is printed only after its *preceding* device work drained
cleanly, so **the last tag in the log is the culprit** — everything it covers
completed; the work launched right after that line is what broke. Grep the log
for ``[op-trace`` and look at the final line.
"""

from __future__ import annotations

import os

import torch

_ENABLED = os.environ.get("SGLANG_DSV4_OP_TRACE", "0") == "1"
_ACCEPT_ENABLED = os.environ.get("SGLANG_DSPARK_ACCEPT_TRACE", "0") == "1"
_MAX_RANGE_ELEMS = 1 << 22  # skip min/max on huge tensors (3D KV pages etc.)

_seq = 0
_last = "step-begin"


def enabled() -> bool:
    return _ENABLED


def _npu():
    return getattr(torch, "npu", None)


def _capturing() -> bool:
    """True while the current stream is being captured (syncing is illegal there).

    ``torch.cuda.is_current_stream_capturing`` is the API this repo uses
    everywhere; on Ascend it goes through torch_npu's transfer_to_npu remap, so
    it reports NPU capture state too. ``torch.npu.*`` is the fallback.
    """
    for getter in (
        lambda: torch.cuda.is_current_stream_capturing(),
        lambda: _npu().is_current_stream_capturing(),
    ):
        try:
            return bool(getter())
        except Exception:
            continue
    return False  # no capture API at all -> not capturing, syncing is safe


def _sync() -> None:
    npu = _npu()
    if npu is not None:
        npu.synchronize()


def _fmt(value) -> str:
    if not isinstance(value, torch.Tensor):
        return repr(value)
    head = (
        f"shape={tuple(value.shape)} dtype={value.dtype} stride={value.stride()} "
        f"contig={value.is_contiguous()} dev={value.device} "
        f"ptr=0x{value.data_ptr():x} nbytes={value.numel() * value.element_size()}"
    )
    if value.numel() == 0:
        return head + " <empty>"
    if value.numel() > _MAX_RANGE_ELEMS:
        return head  # range read on a huge buffer costs more than it tells us
    try:
        return f"{head} min={int(value.min())} max={int(value.max())}"
    except Exception as exc:  # pragma: no cover - defensive
        return f"{head} <range failed: {exc}>"


def trace_op(tag: str, **tensors) -> None:
    """Mark the launch of a device op, after draining the stream.

    The sync is the whole point: if an earlier kernel faulted asynchronously it
    raises here, and the marker printed just before it names the culprit.
    """
    global _seq, _last
    if not _ENABLED:
        return
    capturing = _capturing()
    if not capturing:
        _sync()
    _seq += 1
    print(f"[op-trace {_seq:04d}] {tag}  (after {_last}, capture={capturing})", flush=True)
    for name, value in tensors.items():
        print(f"    {name}: {_fmt(value)}", flush=True)
    _last = tag


def trace_accept(tag: str, **info) -> None:
    """Accept-rate forensics for DSpark (``SGLANG_DSPARK_ACCEPT_TRACE=1``).

    Separate from the op trace on purpose: this one is per-step, not per-op, so
    it is cheap enough to leave on for a whole run. Values are pulled to the host
    (``.tolist()``) by the caller, which also drains the stream, so a run with
    zero acceptance still ends cleanly instead of hiding a device fault.

    Read it row by row: ``target_argmax[i]`` is what the target would pick after
    the i-th verify token, so acceptance needs ``target_argmax[i] ==
    verify_ids[i+1]``. All-zero acceptance with plausible ids means the
    comparison/target side is broken; ids that look like noise means the draft
    side is.
    """
    if not _ACCEPT_ENABLED:
        return
    print(f"[accept-trace] {tag}", flush=True)
    for name, value in info.items():
        print(f"    {name} = {value}", flush=True)


def trace_stage(tag: str, **info) -> None:
    """Coarse-grained marker (step boundaries, forwards, graph replays)."""
    global _seq, _last
    if not _ENABLED:
        return
    capturing = _capturing()
    if not capturing:
        _sync()
    _seq += 1
    extra = "".join(f" {k}={v!r}" for k, v in info.items())
    print(
        f"[op-trace {_seq:04d}] {tag}  (after {_last}, capture={capturing}){extra}",
        flush=True,
    )
    _last = tag
