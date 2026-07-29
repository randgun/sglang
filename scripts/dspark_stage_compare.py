#!/usr/bin/env python3
"""Capture and compare one complete DeepSeek-V4 DSpark speculative round.

This utility deliberately instruments the real SGLang draft model instead of
reimplementing DSpark in a small reference model.  That keeps attention cache
metadata, MoE routing, HC blocks, TP/EP communication, and quantized kernels
identical to serving.

Capture (arguments after ``--`` are normal ``sglang.launch_server`` args):

    python scripts/dspark_stage_compare.py capture \
      --dump-dir /tmp/dspark_gpu \
      --rid-prefix dspark-stage-compare \
      -- \
      --model-path /path/to/target \
      --speculative-algorithm DSPARK \
      --speculative-draft-model-path /path/to/draft \
      ...

Send a greedy request whose rid starts with the selected prefix.  Repeat on
NPU with the same prompt, token IDs, checkpoints, and topology.

Compare:

    python scripts/dspark_stage_compare.py compare \
      --reference /tmp/dspark_gpu \
      --candidate /tmp/dspark_npu \
      --rank 0 \
      --rid dspark-stage-compare-001

The capture directory is organized as:

    rank0/<rid>/call0000/
      model_input.pt
      positions.pt
      input_ids.pt
      stage0/input.pt
      stage0/attn_input.pt
      stage0/attn_output.pt
      stage0/moe_input.pt
      stage0/moe_output.pt
      stage0/output.pt
      ...
      raw_hidden.pt
      x_post_hc.pt
      base_logits.pt
      corrected_logits.pt
      draft_tokens.pt

    rank0/<rid>/target_prefill0000/
      embedding.pt
      layer000.pt
      layer005.pt
      ...
      layer040.pt
      layer042.pt
      target_next_token_logits.pt
      target_sampled_token_ids.pt

    rank0/<rid>/verify0000/
      verify_ids_2d.pt
      forward_input_ids.pt
      forward_positions.pt
      target_verify_logits.pt
      target_predict.pt
      matches.pt
      correct_len.pt
      bonus.pt
      commit_lens.pt
      out_tokens.pt
      target_verify_hidden.pt
      commit_complete.json

Only tensors belonging to a matching request ID are saved, so server warmup
and the launch-time generation request are ignored.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import os
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


_ENV_DUMP_DIR = "SGLANG_DSPARK_STAGE_CAPTURE_DIR"
_ENV_RID_PREFIX = "SGLANG_DSPARK_STAGE_CAPTURE_RID_PREFIX"
_ENV_MAX_CALLS = "SGLANG_DSPARK_STAGE_CAPTURE_MAX_CALLS"
_ENV_EXIT_AFTER_DRAFT = "SGLANG_DSPARK_STAGE_CAPTURE_EXIT_AFTER_DRAFT"
_ENV_EXIT_AFTER_ROUND = "SGLANG_DSPARK_STAGE_CAPTURE_EXIT_AFTER_ROUND"
_ENV_FIXED_TARGET_HIDDEN = "SGLANG_DSPARK_STAGE_CAPTURE_FIXED_TARGET_HIDDEN"
_ENV_FIXED_TARGET_HIDDEN_SEED = (
    "SGLANG_DSPARK_STAGE_CAPTURE_FIXED_TARGET_HIDDEN_SEED"
)
_CAPTURE_INSTALLED = False
_CALL_COUNTS: dict[tuple[int, str], int] = defaultdict(int)
_KV_COUNTS: dict[tuple[int, str], int] = defaultdict(int)
_TARGET_PREFILL_COUNTS: dict[tuple[int, str], int] = defaultdict(int)
_VERIFY_COUNTS: dict[tuple[int, str], int] = defaultdict(int)
_ACTIVE_REQUEST_IDS: dict[int, str] = {}
_ACTIVE_PHASES: dict[int, str] = {}
_ACTIVE_VERIFY_PATHS: dict[int, Path] = {}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _rank() -> int:
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_rank)
    except Exception:
        return int(os.environ.get("RANK", "0"))


def _request_id(model: Any) -> Optional[str]:
    stages = getattr(model, "stages", None)
    if stages:
        # NPU development branches may propagate the request ID directly to
        # DSparkAttention. Upstream main currently does not.
        rid = getattr(stages[0].self_attn, "_debug_probe_key", None)
        if rid is not None:
            return str(rid)
    return _ACTIVE_REQUEST_IDS.get(_rank())


def _batch_request_id(batch: Any) -> Optional[str]:
    reqs = getattr(batch, "reqs", None)
    if not reqs:
        return None
    rid = getattr(reqs[0], "rid", None)
    return str(rid) if rid is not None else None


def _set_active_request(
    batch: Any, phase: str
) -> tuple[int, Optional[str], Optional[str]]:
    rank = _rank()
    previous_rid = _ACTIVE_REQUEST_IDS.get(rank)
    previous_phase = _ACTIVE_PHASES.get(rank)
    rid = _batch_request_id(batch)
    if rid is None:
        _ACTIVE_REQUEST_IDS.pop(rank, None)
    else:
        _ACTIVE_REQUEST_IDS[rank] = rid
    _ACTIVE_PHASES[rank] = phase
    return rank, previous_rid, previous_phase


def _restore_active_request(
    state: tuple[int, Optional[str], Optional[str]],
) -> None:
    rank, previous_rid, previous_phase = state
    if previous_rid is None:
        _ACTIVE_REQUEST_IDS.pop(rank, None)
    else:
        _ACTIVE_REQUEST_IDS[rank] = previous_rid
    if previous_phase is None:
        _ACTIVE_PHASES.pop(rank, None)
    else:
        _ACTIVE_PHASES[rank] = previous_phase


def _capture_enabled(rid: Optional[str]) -> bool:
    if not rid:
        return False
    prefix = os.environ.get(_ENV_RID_PREFIX, "dspark-stage-compare")
    return rid.startswith(prefix)


def _extract_tensor(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _save_tensor(path: Path, value: Any) -> None:
    import torch

    tensor = _extract_tensor(value)
    if tensor is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clone after moving to CPU so later in-place kernels cannot mutate the
    # diagnostic tensor.  torch.save is intentionally synchronous here.
    torch.save(tensor.detach().to("cpu").contiguous().clone(), path)


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _fixed_target_hidden_like(reference):
    """Build device-independent BF16 test data without backend RNG kernels."""
    import torch

    seed = int(os.environ.get(_ENV_FIXED_TARGET_HIDDEN_SEED, "20260723"))
    numel = reference.numel()
    indices = torch.arange(numel, dtype=torch.int64, device="cpu")
    # Integer arithmetic is bit-for-bit stable across PyTorch/device versions.
    values = ((indices * 48271 + seed * 69621) % 65521).to(torch.float32)
    values = values.div_(32760.0).sub_(1.0).to(torch.bfloat16)
    return values.reshape(reference.shape).to(device=reference.device)


def _norm_rope_reference(kv, *, weight, eps: float, freqs_cis, positions):
    """Canonical BF16 RMSNorm + interleaved RoPE reference."""
    import torch

    normalized = kv.detach().float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + float(eps)
    )
    normalized = (normalized * weight.detach().float()).to(kv.dtype)

    rope_dim = int(freqs_cis.shape[-1]) * 2
    result = normalized.clone()
    rope = normalized[..., -rope_dim:].float().unflatten(-1, (-1, 2))
    positions = positions.to(device=freqs_cis.device, dtype=torch.int64)
    freqs_real = freqs_cis.real.contiguous()[positions].float()
    freqs_imag = freqs_cis.imag.contiguous()[positions].float()
    real, imag = rope[..., 0], rope[..., 1]
    rotated = torch.stack(
        (
            real * freqs_real - imag * freqs_imag,
            real * freqs_imag + imag * freqs_real,
        ),
        dim=-1,
    ).flatten(-2)
    result[..., -rope_dim:] = rotated.to(result.dtype)
    return result


def _read_stored_swa_canonical(pool, *, layer_id: int, swa_loc):
    """Read logical SWA rows as dequantized [tokens, 512] values."""
    import torch

    valid_locs = swa_loc[swa_loc >= 0].to(torch.int64)
    if valid_locs.numel() == 0:
        return torch.empty((0, 512), dtype=torch.float32)

    # Ascend exposes a canonical PA_ND BF16 accessor.
    get_swa_buffer = getattr(pool, "get_swa_buffer", None)
    if get_swa_buffer is not None:
        stored = get_swa_buffer(layer_id, valid_locs)
        if stored.ndim == 3 and stored.shape[-2] == 1:
            stored = stored.squeeze(-2)
        return stored.detach().float().cpu()

    # CUDA stores 448 FP8 values, 64 BF16 RoPE values, and seven UE8M0
    # scales in a packed uint8 page. Decode it to the same canonical layout.
    raw = pool.get_swa_raw_buffer(layer_id).detach().cpu()
    if raw.dtype != torch.uint8 or raw.ndim != 2:
        raise RuntimeError(
            "unsupported SWA storage layout for canonical capture: "
            f"shape={tuple(raw.shape)}, dtype={raw.dtype}"
        )
    try:
        # Current tree.
        from sglang.kernels.ops.attention.dsv4.quant_k_cache import fp8_dtype
    except ModuleNotFoundError:
        # Upstream main before the kernel-package move.
        from sglang.srt.layers.attention.dsv4.quant_k_cache import fp8_dtype

    page_size = int(pool.swa_kv_pool.page_size)
    data_bytes_per_token = 448 + 64 * 2
    scale_bytes_per_token = 8
    scale_base = page_size * data_bytes_per_token
    rows = []
    for loc in valid_locs.detach().cpu().tolist():
        page, offset = divmod(int(loc), page_size)
        page_row = raw[page]
        data_base = offset * data_bytes_per_token
        nope = (
            page_row[data_base : data_base + 448]
            .contiguous()
            .view(fp8_dtype)
            .float()
            .reshape(7, 64)
        )
        rope = (
            page_row[data_base + 448 : data_base + data_bytes_per_token]
            .contiguous()
            .view(torch.bfloat16)
            .float()
        )
        scale_offset = scale_base + offset * scale_bytes_per_token
        exponents = page_row[scale_offset : scale_offset + 7].to(torch.int32)
        scales = torch.pow(2.0, exponents.float() - 127.0)
        rows.append(torch.cat(((nope * scales[:, None]).reshape(-1), rope)))
    return torch.stack(rows)


def _new_call_dir(rid: str) -> Optional[Path]:
    rank = _rank()
    key = (rank, rid)
    call_id = _CALL_COUNTS[key]
    max_calls = int(os.environ.get(_ENV_MAX_CALLS, "8"))
    if call_id >= max_calls:
        return None
    _CALL_COUNTS[key] += 1
    root = Path(os.environ[_ENV_DUMP_DIR])
    return root / f"rank{rank}" / _safe_name(rid) / f"call{call_id:04d}"


def _new_verify_dir(rid: str) -> Optional[Path]:
    rank = _rank()
    key = (rank, rid)
    verify_id = _VERIFY_COUNTS[key]
    max_calls = int(os.environ.get(_ENV_MAX_CALLS, "8"))
    if verify_id >= max_calls:
        return None
    _VERIFY_COUNTS[key] += 1
    root = Path(os.environ[_ENV_DUMP_DIR])
    return root / f"rank{rank}" / _safe_name(rid) / f"verify{verify_id:04d}"


def _install_target_prefill_hooks(worker: Any, batch: Any, rid: str):
    """Capture canonical target hidden states at embedding and 5-layer intervals."""
    rank = _rank()
    key = (rank, rid)
    capture_id = _TARGET_PREFILL_COUNTS[key]
    max_calls = int(os.environ.get(_ENV_MAX_CALLS, "8"))
    if capture_id >= max_calls:
        return [], None
    _TARGET_PREFILL_COUNTS[key] += 1

    target_model = worker.target_worker.model_runner.model
    target_core = getattr(target_model, "model", None)
    layers = getattr(target_core, "layers", None)
    if target_core is None or layers is None:
        raise RuntimeError(
            "DSpark target capture expected target_model.model.layers, got "
            f"{type(target_model).__name__}"
        )

    start_layer = int(getattr(target_core, "start_layer", 0))
    end_layer = int(getattr(target_core, "end_layer", len(layers)))
    selected_layers = [
        layer_id
        for layer_id in range(start_layer, end_layer)
        if layer_id % 5 == 0
    ]
    if end_layer > start_layer and end_layer - 1 not in selected_layers:
        selected_layers.append(end_layer - 1)

    root = Path(os.environ[_ENV_DUMP_DIR])
    path = (
        root
        / f"rank{rank}"
        / _safe_name(rid)
        / f"target_prefill{capture_id:04d}"
    )
    _save_tensor(path / "input_ids.pt", getattr(batch, "input_ids", None))
    _save_tensor(path / "seq_lens.pt", getattr(batch, "seq_lens", None))
    batch_input_ids = getattr(batch, "input_ids", None)
    num_real_tokens = (
        int(batch_input_ids.numel()) if batch_input_ids is not None else None
    )

    def canonical_rows(tensor):
        # NPU DP/EP execution may pad the target prefill batch (for example,
        # 9 real tokens -> 16 internal rows). TargetHiddenKvInjector applies
        # the same leading-row trim before DSpark injection.
        if (
            tensor is not None
            and num_real_tokens is not None
            and tensor.ndim >= 1
            and tensor.shape[0] > num_real_tokens
        ):
            return tensor[:num_real_tokens]
        return tensor

    handles = []
    embed_tokens = getattr(target_core, "embed_tokens", None)
    if start_layer == 0 and embed_tokens is not None:

        def embedding_hook(module, args, output):
            del module
            if args:
                _save_tensor(
                    path / "embedding_input_ids.pt", canonical_rows(args[0])
                )
            _save_tensor(path / "embedding.pt", canonical_rows(output))

        handles.append(embed_tokens.register_forward_hook(embedding_hook))

    use_fused_mhc = bool(
        getattr(target_core, "use_fused_mhc_post_pre", False)
    )
    for layer_id in selected_layers:
        layer = layers[layer_id]

        def layer_hook(module, args, output, *, captured_layer_id=layer_id):
            del args
            completed = None
            if isinstance(output, (tuple, list)) and output:
                hidden_states = output[0]
                if use_fused_mhc and len(output) >= 4:
                    completed = module.hc_post(
                        hidden_states, output[1], output[2], output[3]
                    )
                else:
                    completed = hidden_states
            else:
                completed = _extract_tensor(output)
            if completed is not None and completed.ndim == 3:
                # Match DeepSeek-V4's DSpark aux-hidden capture semantics.
                completed = completed.mean(dim=1)
            _save_tensor(
                path / f"layer{captured_layer_id:03d}.pt",
                canonical_rows(completed),
            )

        handles.append(layer.register_forward_hook(layer_hook))

    _save_json(
        path / "meta.json",
        {
            "rid": rid,
            "rank": rank,
            "phase": "prefill",
            "start_layer": start_layer,
            "end_layer": end_layer,
            "selected_layers": selected_layers,
            "num_real_tokens": num_real_tokens,
            "capture_semantics": (
                "embedding output; completed layer hidden after hc_post when "
                "fused MHC is enabled; mean over hc_mult; trim internal padded "
                "rows to ScheduleBatch.input_ids.numel()"
            ),
        },
    )
    return handles, path


def _install_capture_patch() -> None:
    global _CAPTURE_INSTALLED
    if _CAPTURE_INSTALLED:
        return

    from sglang.srt.models.deepseek_v4_dspark import (
        CommitKvProj,
        DSparkV4MarkovHead,
        DeepseekV4ForCausalLMDSpark,
    )
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
        DSparkWorkerV2,
    )
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        TargetVerifyExecutor,
    )

    if (
        getattr(DeepseekV4ForCausalLMDSpark, "_stage_compare_patched", False)
        and getattr(DSparkWorkerV2, "_stage_compare_patched", False)
    ):
        _CAPTURE_INSTALLED = True
        return

    original_forward = DeepseekV4ForCausalLMDSpark.forward
    original_compute_base_logits = DeepseekV4ForCausalLMDSpark.compute_base_logits
    original_write_target_hidden_kv = (
        DeepseekV4ForCausalLMDSpark.write_target_hidden_kv
    )
    original_sample_block = DSparkV4MarkovHead.sample_block
    original_forward_prefill = DSparkWorkerV2._forward_prefill
    original_forward_decode = DSparkWorkerV2._forward_decode
    original_run_non_compact = TargetVerifyExecutor.run_non_compact
    original_run_compact = TargetVerifyExecutor.run_compact
    original_forward_prepared_verify = (
        TargetVerifyExecutor._forward_prepared_verify
    )
    original_accept_and_finalize = TargetVerifyExecutor.accept_and_finalize
    original_commit_hidden = TargetVerifyExecutor.commit_hidden
    write_target_hidden_parameters = inspect.signature(
        original_write_target_hidden_kv
    ).parameters

    def captured_forward_prefill(self, batch, *args, **kwargs):
        state = _set_active_request(batch, "prefill")
        rid = _batch_request_id(batch)
        handles = []
        capture_path = None
        try:
            if _capture_enabled(rid):
                handles, capture_path = _install_target_prefill_hooks(
                    self, batch, rid
                )
            result = original_forward_prefill(self, batch, *args, **kwargs)
            if capture_path is not None:
                logits_output = getattr(result, "logits_output", None)
                _save_tensor(
                    capture_path / "target_next_token_logits.pt",
                    getattr(logits_output, "next_token_logits", None),
                )
                _save_tensor(
                    capture_path / "target_sampled_token_ids.pt",
                    getattr(result, "next_token_ids", None),
                )
            return result
        finally:
            for handle in handles:
                handle.remove()
            _restore_active_request(state)

    def captured_forward_decode(self, batch, *args, **kwargs):
        state = _set_active_request(batch, "decode")
        rank = _rank()
        rid = _batch_request_id(batch)
        verify_path = _new_verify_dir(rid) if _capture_enabled(rid) else None
        if verify_path is not None:
            _ACTIVE_VERIFY_PATHS[rank] = verify_path
            _save_tensor(verify_path / "prefix_lens.pt", getattr(batch, "seq_lens", None))
            _save_tensor(
                verify_path / "req_pool_indices.pt",
                getattr(batch, "req_pool_indices", None),
            )
            _save_json(
                verify_path / "meta.json",
                {
                    "rid": rid,
                    "rank": rank,
                    "phase": "decode_round",
                    "source": "sglang",
                },
            )
        try:
            result = original_forward_decode(self, batch, *args, **kwargs)
            if verify_path is not None:
                _save_tensor(
                    verify_path / "round_next_token_ids.pt",
                    getattr(result, "next_token_ids", None),
                )
                _save_tensor(
                    verify_path / "round_accept_lens.pt",
                    getattr(result, "accept_lens", None),
                )
                _save_tensor(
                    verify_path / "round_block_accept_lens.pt",
                    getattr(result, "block_accept_lens", None),
                )
                _save_tensor(
                    verify_path / "round_new_seq_lens.pt",
                    getattr(result, "new_seq_lens", None),
                )
                next_draft = getattr(result, "next_draft_input", None)
                _save_tensor(
                    verify_path / "next_draft_bonus_tokens.pt",
                    getattr(next_draft, "bonus_tokens", None),
                )
                _save_tensor(
                    verify_path / "next_draft_new_seq_lens.pt",
                    getattr(next_draft, "new_seq_lens", None),
                )
                _save_json(
                    verify_path / "round_complete.json",
                    {"complete": True, "rid": rid, "rank": rank},
                )
                if os.environ.get(_ENV_EXIT_AFTER_ROUND) == "1":
                    from sglang.srt.distributed import get_tp_group

                    get_tp_group().barrier()
                    raise RuntimeError(
                        "DSpark full-round capture complete after commit "
                        f"(rid={rid!r}, rank={rank})"
                    )
            return result
        finally:
            if _ACTIVE_VERIFY_PATHS.get(rank) == verify_path:
                _ACTIVE_VERIFY_PATHS.pop(rank, None)
            _restore_active_request(state)

    def _save_verify_result(path: Optional[Path], result: Any) -> None:
        if path is None or result is None:
            return
        logits_output = getattr(result, "logits_output", None)
        logits = getattr(logits_output, "next_token_logits", None)
        hidden = getattr(logits_output, "hidden_states", None)
        _save_tensor(path / "target_verify_logits_full.pt", logits)
        _save_tensor(path / "target_verify_hidden.pt", hidden)
        if logits is not None:
            _save_tensor(path / "target_verify_predict_full.pt", logits.argmax(dim=-1))
        _save_json(
            path / "verify_forward_result.json",
            {
                "can_run_graph": bool(getattr(result, "can_run_cuda_graph", False)),
            },
        )

    def captured_run_non_compact(
        self,
        *,
        batch,
        draft_input,
        verify_ids_2d,
        verify_window,
        sampling_info,
    ):
        path = _ACTIVE_VERIFY_PATHS.get(_rank())
        if path is not None:
            _save_tensor(path / "verify_ids_2d.pt", verify_ids_2d)
            _save_tensor(path / "draft_tokens.pt", verify_ids_2d[:, 1:])
            _save_tensor(path / "positions_2d.pt", verify_window.positions_2d)
            _save_tensor(path / "verify_cache_loc.pt", verify_window.verify_cache_loc)
            _save_tensor(
                path / "verify_cache_loc_2d.pt",
                verify_window.verify_cache_loc_2d,
            )
            _save_json(path / "verify_mode.json", {"mode": "non_compact"})
        result = original_run_non_compact(
            self,
            batch=batch,
            draft_input=draft_input,
            verify_ids_2d=verify_ids_2d,
            verify_window=verify_window,
            sampling_info=sampling_info,
        )
        _save_verify_result(path, result)
        return result

    def captured_run_compact(
        self,
        *,
        batch,
        layout,
        draft_block_ids,
        draft_tokens,
        bs,
        device,
        sampling_info,
        inject_gate=False,
    ):
        path = _ACTIVE_VERIFY_PATHS.get(_rank())
        if path is not None:
            _save_tensor(path / "anchor_tokens.pt", draft_block_ids[:, :1])
            _save_tensor(path / "draft_tokens.pt", draft_tokens)
            _save_tensor(path / "layout_verify_lens.pt", getattr(layout, "verify_lens", None))
            _save_tensor(path / "layout_qo_indptr.pt", getattr(layout, "qo_indptr", None))
            _save_json(path / "verify_mode.json", {"mode": "compact"})
        result, hidden_strided = original_run_compact(
            self,
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            sampling_info=sampling_info,
            inject_gate=inject_gate,
        )
        _save_verify_result(path, result)
        if path is not None:
            _save_tensor(path / "target_verify_hidden_strided.pt", hidden_strided)
        return result, hidden_strided

    def captured_forward_prepared_verify(
        self,
        *,
        batch,
        verify_input,
        seq_lens_cpu_backup,
        seq_lens_sum_backup,
    ):
        path = _ACTIVE_VERIFY_PATHS.get(_rank())
        if path is not None:
            _save_tensor(path / "forward_input_ids.pt", getattr(verify_input, "draft_token", None))
            _save_tensor(path / "forward_positions.pt", getattr(verify_input, "positions", None))
            _save_tensor(path / "forward_out_cache_loc.pt", getattr(batch, "out_cache_loc", None))
            _save_tensor(path / "forward_seq_lens.pt", getattr(batch, "seq_lens", None))
            _save_tensor(
                path / "forward_req_pool_indices.pt",
                getattr(batch, "req_pool_indices", None),
            )
        return original_forward_prepared_verify(
            self,
            batch=batch,
            verify_input=verify_input,
            seq_lens_cpu_backup=seq_lens_cpu_backup,
            seq_lens_sum_backup=seq_lens_sum_backup,
        )

    def captured_accept_and_finalize(
        self,
        *,
        folded_accept,
        bs,
        verify_ids_2d,
        target_logits,
        draft_block,
        sampling_info,
        draft_input,
        layout,
        prefix_lens,
        draft_tokens,
    ):
        import torch

        path = _ACTIVE_VERIFY_PATHS.get(_rank())
        if path is not None:
            _save_tensor(path / "verify_ids_2d.pt", verify_ids_2d)
            _save_tensor(path / "draft_tokens.pt", draft_tokens)
            _save_tensor(path / "prefix_lens.pt", prefix_lens)
            if target_logits is not None:
                logits_3d = target_logits.view(bs, self.verify_num_draft_tokens, -1)
                _save_tensor(
                    path / "target_verify_logits.pt",
                    logits_3d[:, :-1].reshape(-1, logits_3d.shape[-1]),
                )
                _save_tensor(path / "bonus_logits.pt", logits_3d[:, -1])
                predicts = logits_3d.argmax(dim=-1)
                _save_tensor(path / "target_predict_full.pt", predicts)
                _save_tensor(path / "target_predict.pt", predicts[:, :-1])
                _save_tensor(path / "bonus_predict.pt", predicts[:, -1])
                _save_tensor(
                    path / "matches.pt",
                    verify_ids_2d[:, 1:] == predicts[:, :-1],
                )
        result = original_accept_and_finalize(
            self,
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )
        if path is not None:
            for name in (
                "correct_len",
                "bonus",
                "cap_trim_lens",
                "commit_lens",
                "new_seq_lens",
                "out_tokens",
            ):
                _save_tensor(path / f"{name}.pt", getattr(result, name, None))
            out_tokens = result.out_tokens.view(bs, self.verify_num_draft_tokens)
            max_commit = int(result.commit_lens.max().item())
            committed = torch.full(
                (bs, max_commit),
                -1,
                dtype=out_tokens.dtype,
                device=out_tokens.device,
            )
            for req_id in range(bs):
                commit_len = int(result.commit_lens[req_id].item())
                committed[req_id, :commit_len] = out_tokens[req_id, :commit_len]
            _save_tensor(path / "committed_token_ids.pt", committed)
            _save_json(
                path / "accept_complete.json",
                {
                    "folded_accept": bool(folded_accept),
                    "batch_size": int(bs),
                },
            )
            if (
                folded_accept
                and self.verify_epilogue is not None
                and self.verify_epilogue.folds_commit
            ):
                _save_json(
                    path / "commit_complete.json",
                    {
                        "complete": True,
                        "compact": True,
                        "folded_into_verify_graph": True,
                        "batch_size": int(bs),
                    },
                )
        return result

    def captured_commit_hidden(
        self,
        *,
        batch,
        layout,
        hidden_strided,
        verify_window,
        logits_output,
        commit_lens,
        bs,
        run_compact,
    ):
        path = _ACTIVE_VERIFY_PATHS.get(_rank())
        if path is not None:
            _save_tensor(path / "commit_lens.pt", commit_lens)
            _save_tensor(path / "commit_hidden_strided.pt", hidden_strided)
            _save_tensor(
                path / "commit_hidden_full.pt",
                getattr(logits_output, "hidden_states", None),
            )
            if verify_window is not None:
                _save_tensor(
                    path / "commit_positions_2d.pt",
                    getattr(verify_window, "positions_2d", None),
                )
                _save_tensor(
                    path / "commit_cache_loc_2d.pt",
                    getattr(verify_window, "verify_cache_loc_2d", None),
                )
        result = original_commit_hidden(
            self,
            batch=batch,
            layout=layout,
            hidden_strided=hidden_strided,
            verify_window=verify_window,
            logits_output=logits_output,
            commit_lens=commit_lens,
            bs=bs,
            run_compact=run_compact,
        )
        if path is not None:
            _save_json(
                path / "commit_complete.json",
                {"complete": True, "compact": bool(run_compact), "batch_size": int(bs)},
            )
        return result

    def captured_forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        rid = _request_id(self)
        if not _capture_enabled(rid):
            return original_forward(
                self,
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        call_dir = _new_call_dir(rid)
        if call_dir is None:
            return original_forward(
                self,
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        handles = []
        _save_tensor(call_dir / "input_ids.pt", input_ids)
        _save_tensor(call_dir / "positions.pt", positions)
        if input_embeds is not None:
            _save_tensor(call_dir / "model_input.pt", input_embeds)

        for stage_id, stage in enumerate(self.stages):
            stage_dir = call_dir / f"stage{stage_id}"

            def stage_pre_hook(module, args, *, path=stage_dir):
                del module
                # DSparkV4Stage.forward(positions, hidden_states, forward_batch)
                if len(args) >= 2:
                    _save_tensor(path / "input.pt", args[1])

            def stage_hook(module, args, output, *, path=stage_dir):
                del module, args
                _save_tensor(path / "output.pt", output)

            def attn_pre_hook(module, args, *, path=stage_dir):
                del module
                if len(args) >= 2:
                    _save_tensor(path / "attn_input.pt", args[1])

            def attn_hook(module, args, output, *, path=stage_dir):
                del module, args
                _save_tensor(path / "attn_output.pt", output)

            def moe_pre_hook(module, args, *, path=stage_dir):
                del module
                if args:
                    _save_tensor(path / "moe_input.pt", args[0])

            def moe_hook(module, args, output, *, path=stage_dir):
                del module, args
                _save_tensor(path / "moe_output.pt", output)

            handles.append(stage.register_forward_pre_hook(stage_pre_hook))
            handles.append(stage.register_forward_hook(stage_hook))
            handles.append(stage.self_attn.register_forward_pre_hook(attn_pre_hook))
            handles.append(stage.self_attn.register_forward_hook(attn_hook))
            # The exact MoE implementation varies with the backend, but the
            # decoder layer consistently exposes it as ``mlp``.
            if hasattr(stage, "mlp"):
                handles.append(stage.mlp.register_forward_pre_hook(moe_pre_hook))
                handles.append(stage.mlp.register_forward_hook(moe_hook))

        try:
            output = original_forward(
                self,
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        finally:
            for handle in handles:
                handle.remove()

        hidden = getattr(output, "hidden_states", None)
        _save_tensor(call_dir / "raw_hidden.pt", hidden)
        self._stage_compare_last_call_dir = call_dir
        self.markov_head._stage_compare_owner = self
        _save_json(
            call_dir / "meta.json",
            {
                "rid": rid,
                "rank": _rank(),
                "num_stages": len(self.stages),
                "input_ids_shape": list(input_ids.shape),
                "positions_shape": list(positions.shape),
            },
        )
        return output

    def captured_compute_base_logits(self, x):
        result = original_compute_base_logits(self, x)
        call_dir = getattr(self, "_stage_compare_last_call_dir", None)
        if call_dir is not None:
            base_logits, x_post_hc = result
            _save_tensor(Path(call_dir) / "x_post_hc.pt", x_post_hc)
            _save_tensor(Path(call_dir) / "base_logits.pt", base_logits)
        return result

    def captured_sample_block(
        self,
        base_logits,
        *,
        first_prev_tokens,
        hidden_states,
        sampler,
    ):
        result = original_sample_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )
        owner = getattr(self, "_stage_compare_owner", None)
        call_dir = getattr(owner, "_stage_compare_last_call_dir", None)
        if call_dir is not None:
            draft_tokens, corrected_logits = result
            _save_tensor(Path(call_dir) / "anchor_tokens.pt", first_prev_tokens)
            _save_tensor(Path(call_dir) / "draft_tokens.pt", draft_tokens)
            _save_tensor(Path(call_dir) / "corrected_logits.pt", corrected_logits)
            if os.environ.get(_ENV_EXIT_AFTER_DRAFT) == "1":
                # Ensure every TP rank has flushed its files before the first
                # scheduler exception makes the launcher tear down the worker
                # process tree.
                from sglang.srt.distributed import get_tp_group

                get_tp_group().barrier()
                raise RuntimeError(
                    "DSpark draft capture complete; stopping before target verify "
                    f"(rid={_request_id(owner)!r}, rank={_rank()})"
                )
        return result

    def captured_write_target_hidden_kv(
        self,
        *,
        main_hidden,
        swa_loc,
        positions,
        pool,
        probe_key=None,
        probe_phase="unknown",
    ):
        if probe_key is None:
            probe_key = _ACTIVE_REQUEST_IDS.get(_rank())
        if probe_phase == "unknown":
            probe_phase = _ACTIVE_PHASES.get(_rank(), probe_phase)
        original_main_hidden = main_hidden
        fixed_enabled = os.environ.get(_ENV_FIXED_TARGET_HIDDEN) == "1"
        if fixed_enabled:
            main_hidden = _fixed_target_hidden_like(main_hidden)

        capture_path = None
        main_x = None
        kvs = None
        if _capture_enabled(probe_key):
            rank = _rank()
            key = (rank, str(probe_key))
            capture_id = _KV_COUNTS[key]
            _KV_COUNTS[key] += 1
            root = Path(os.environ[_ENV_DUMP_DIR])
            path = (
                root
                / f"rank{rank}"
                / _safe_name(str(probe_key))
                / f"kv_inject{capture_id:04d}"
            )
            capture_path = path
            if fixed_enabled:
                _save_tensor(path / "target_hidden_original.pt", original_main_hidden)
            _save_tensor(path / "target_hidden.pt", main_hidden)
            main_x = self.project_target_hidden(main_hidden)
            _save_tensor(path / "main_x.pt", main_x)
            _save_tensor(path / "positions.pt", positions)
            _save_tensor(path / "swa_loc.pt", swa_loc)

            kvs = CommitKvProj.execute(
                main_x=main_x,
                wkv_linears=[stage.self_attn.wkv for stage in self.stages],
            )
            valid = swa_loc >= 0
            _save_tensor(path / "valid_positions.pt", positions[valid])
            for stage, kv in zip(self.stages, kvs):
                stage_path = path / f"stage{stage.stage_id}"
                _save_tensor(stage_path / "raw_kv.pt", kv)
                expected = _norm_rope_reference(
                    kv,
                    weight=stage.self_attn.kv_norm.weight,
                    eps=stage.self_attn.eps,
                    freqs_cis=stage.self_attn.freqs_cis,
                    positions=positions,
                )
                _save_tensor(stage_path / "norm_rope_reference.pt", expected)
                _save_tensor(
                    stage_path / "expected_stored_kv.pt", expected[valid]
                )
            _save_json(
                path / "meta.json",
                {
                    "rid": str(probe_key),
                    "rank": rank,
                    "phase": str(probe_phase),
                    "fixed_target_hidden": fixed_enabled,
                    "fixed_target_hidden_seed": (
                        int(
                            os.environ.get(
                                _ENV_FIXED_TARGET_HIDDEN_SEED, "20260723"
                            )
                        )
                        if fixed_enabled
                        else None
                    ),
                },
            )

        write_kwargs = {
            "main_hidden": main_hidden,
            "swa_loc": swa_loc,
            "positions": positions,
            "pool": pool,
        }
        # Development branches add diagnostic-only arguments. Do not pass
        # them to upstream main, whose public method has the four arguments
        # above.
        if "probe_key" in write_target_hidden_parameters:
            write_kwargs["probe_key"] = probe_key
        if "probe_phase" in write_target_hidden_parameters:
            write_kwargs["probe_phase"] = probe_phase
        result = original_write_target_hidden_kv(self, **write_kwargs)
        if capture_path is not None:
            for stage in self.stages:
                stored = _read_stored_swa_canonical(
                    pool,
                    layer_id=stage.self_attn.layer_id,
                    swa_loc=swa_loc,
                )
                _save_tensor(
                    capture_path / f"stage{stage.stage_id}" / "stored_kv.pt",
                    stored,
                )
        return result

    DeepseekV4ForCausalLMDSpark.forward = captured_forward
    DeepseekV4ForCausalLMDSpark.compute_base_logits = captured_compute_base_logits
    DeepseekV4ForCausalLMDSpark.write_target_hidden_kv = (
        captured_write_target_hidden_kv
    )
    DSparkV4MarkovHead.sample_block = captured_sample_block
    DSparkWorkerV2._forward_prefill = captured_forward_prefill
    DSparkWorkerV2._forward_decode = captured_forward_decode
    TargetVerifyExecutor.run_non_compact = captured_run_non_compact
    TargetVerifyExecutor.run_compact = captured_run_compact
    TargetVerifyExecutor._forward_prepared_verify = captured_forward_prepared_verify
    TargetVerifyExecutor.accept_and_finalize = captured_accept_and_finalize
    TargetVerifyExecutor.commit_hidden = captured_commit_hidden
    DeepseekV4ForCausalLMDSpark._stage_compare_patched = True
    DSparkWorkerV2._stage_compare_patched = True
    _CAPTURE_INSTALLED = True


def _find_request_dir(root: Path, rank: int, rid: Optional[str]) -> Path:
    rank_dir = root / f"rank{rank}"
    if not rank_dir.is_dir():
        raise FileNotFoundError(f"rank directory does not exist: {rank_dir}")
    if rid is not None:
        request_dir = rank_dir / _safe_name(rid)
        if not request_dir.is_dir():
            raise FileNotFoundError(f"request directory does not exist: {request_dir}")
        return request_dir
    candidates = sorted(path for path in rank_dir.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            f"{rank_dir} contains {len(candidates)} request directories; pass --rid"
        )
    return candidates[0]


def _iter_common_tensors(reference: Path, candidate: Path) -> Iterable[Path]:
    reference_files = {
        path.relative_to(reference)
        for path in reference.rglob("*.pt")
        if path.is_file()
    }
    candidate_files = {
        path.relative_to(candidate)
        for path in candidate.rglob("*.pt")
        if path.is_file()
    }
    missing_candidate = sorted(reference_files - candidate_files)
    missing_reference = sorted(candidate_files - reference_files)
    if missing_candidate:
        print("Missing in candidate:")
        for path in missing_candidate:
            print(f"  {path}")
    if missing_reference:
        print("Missing in reference:")
        for path in missing_reference:
            print(f"  {path}")
    def execution_order(path: Path):
        parts = path.parts
        name_order = {
            "target_hidden.pt": 0,
            "main_x.pt": 1,
            "positions.pt": 2,
            "swa_loc.pt": 3,
            "target_next_token_logits.pt": 4,
            "target_sampled_token_ids.pt": 5,
            "input_ids.pt": 10,
            "model_input.pt": 11,
            "input.pt": 20,
            "attn_input.pt": 21,
            "attn_output.pt": 22,
            "moe_input.pt": 23,
            "moe_output.pt": 24,
            "output.pt": 25,
            "raw_hidden.pt": 40,
            "x_post_hc.pt": 41,
            "base_logits.pt": 42,
            "anchor_tokens.pt": 43,
            "corrected_logits.pt": 44,
            "draft_tokens.pt": 45,
            "verify_ids_2d.pt": 50,
            "forward_input_ids.pt": 51,
            "forward_positions.pt": 52,
            "forward_seq_lens.pt": 53,
            "forward_req_pool_indices.pt": 54,
            "forward_out_cache_loc.pt": 55,
            "target_verify_hidden.pt": 56,
            "target_verify_logits_full.pt": 57,
            "target_verify_logits.pt": 58,
            "target_predict_full.pt": 59,
            "target_predict.pt": 60,
            "bonus_logits.pt": 61,
            "bonus_predict.pt": 62,
            "matches.pt": 63,
            "correct_len.pt": 70,
            "bonus.pt": 71,
            "cap_trim_lens.pt": 72,
            "commit_lens.pt": 73,
            "new_seq_lens.pt": 74,
            "out_tokens.pt": 75,
            "commit_hidden_full.pt": 80,
            "commit_hidden_strided.pt": 81,
            "committed_token_ids.pt": 82,
            "round_next_token_ids.pt": 90,
            "round_accept_lens.pt": 91,
            "round_new_seq_lens.pt": 92,
        }
        top = parts[0] if parts else ""
        if top.startswith("target_prefill"):
            group = 0
            sequence = int(top.removeprefix("target_prefill") or 0)
        elif top.startswith("kv_inject"):
            sequence = int(top.removeprefix("kv_inject") or 0)
            group = 1 if sequence == 0 else 4 + (sequence - 1) * 3
        elif top.startswith("call"):
            sequence = int(top.removeprefix("call") or 0)
            group = 2 + sequence * 3
        elif top.startswith("verify"):
            sequence = int(top.removeprefix("verify") or 0)
            group = 3 + sequence * 3
        else:
            group = 10_000
            sequence = 0
        stage = 99
        for part in parts:
            if part.startswith("stage") and part[5:].isdigit():
                stage = int(part[5:])
                break
        return (
            group,
            sequence,
            stage,
            name_order.get(path.name, 100),
            str(path),
        )

    return sorted(reference_files & candidate_files, key=execution_order)


def _compare(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F

    def trim_target_prefill_padding(
        request_root: Path,
        relative: Path,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not relative.parts
            or not relative.parts[0].startswith("target_prefill")
            or tensor.ndim < 1
            or relative.name
            in {
                "input_ids.pt",
                "seq_lens.pt",
                "embedding_input_ids.pt",
            }
        ):
            return tensor
        input_ids_path = request_root / relative.parts[0] / "input_ids.pt"
        if not input_ids_path.is_file():
            return tensor
        input_ids = torch.load(
            input_ids_path, map_location="cpu", weights_only=True
        )
        if not isinstance(input_ids, torch.Tensor):
            return tensor
        num_real_tokens = int(input_ids.numel())
        if tensor.shape[0] > num_real_tokens:
            return tensor[:num_real_tokens]
        return tensor

    def print_target_hidden_cross_cosine(
        relative: Path,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> None:
        if (
            relative.name != "target_hidden.pt"
            or not relative.parent.name.startswith("kv_inject")
            or reference.ndim < 2
            or tuple(reference.shape) != tuple(candidate.shape)
        ):
            return

        num_blocks = 3
        hidden_width = int(reference.shape[-1])
        if hidden_width % num_blocks != 0:
            print(
                "  target-hidden cross cosine skipped: "
                f"last dimension {hidden_width} is not divisible by {num_blocks}"
            )
            return

        block_width = hidden_width // num_blocks
        ref_blocks = [
            reference[..., i * block_width : (i + 1) * block_width]
            .float()
            .reshape(-1)
            for i in range(num_blocks)
        ]
        cand_blocks = [
            candidate[..., i * block_width : (i + 1) * block_width]
            .float()
            .reshape(-1)
            for i in range(num_blocks)
        ]
        matrix = [
            [
                float(
                    F.cosine_similarity(
                        ref_block.reshape(1, -1),
                        cand_block.reshape(1, -1),
                        dim=-1,
                    ).item()
                )
                for cand_block in cand_blocks
            ]
            for ref_block in ref_blocks
        ]

        best_permutation = max(
            itertools.permutations(range(num_blocks)),
            key=lambda permutation: sum(
                matrix[ref_id][permutation[ref_id]]
                for ref_id in range(num_blocks)
            ),
        )
        diagonal_mean = sum(matrix[i][i] for i in range(num_blocks)) / num_blocks
        best_mean = (
            sum(
                matrix[ref_id][best_permutation[ref_id]]
                for ref_id in range(num_blocks)
            )
            / num_blocks
        )
        row_best = [max(row) for row in matrix]

        print(
            "  target-hidden 3x3 cross cosine "
            f"(reference rows -> candidate columns, block_width={block_width}):"
        )
        print("                    candidate[0]  candidate[1]  candidate[2]")
        for ref_id, row in enumerate(matrix):
            print(
                f"    reference[{ref_id}]"
                + "".join(f"  {value:12.8f}" for value in row)
            )
        print(
            "  best mapping: "
            + ", ".join(
                f"reference[{ref_id}]->candidate[{cand_id}]"
                for ref_id, cand_id in enumerate(best_permutation)
            )
        )
        print(
            f"  diagonal_mean={diagonal_mean:.8f}, "
            f"best_permuted_mean={best_mean:.8f}"
        )

        identity = tuple(range(num_blocks))
        if (
            best_permutation != identity
            and best_mean >= 0.80
            and best_mean - diagonal_mean >= 0.05
        ):
            diagnosis = (
                "likely feature/layer ordering mismatch: a non-identity mapping "
                "restores high cosine"
            )
        elif max(row_best) < 0.50:
            diagnosis = (
                "all three feature/layer blocks differ strongly; block ordering "
                "does not explain the mismatch"
            )
        elif diagonal_mean >= 0.80 and best_permutation == identity:
            diagnosis = (
                "feature/layer ordering is consistent; remaining error is within "
                "corresponding blocks"
            )
        else:
            diagnosis = (
                "mixed/ambiguous mismatch; inspect the matrix and source layer "
                "captures before attributing it to ordering"
            )
        print(f"  diagnosis: {diagnosis}")

    reference_root = _find_request_dir(Path(args.reference), args.rank, args.rid)
    candidate_root = _find_request_dir(Path(args.candidate), args.rank, args.rid)
    paths = list(_iter_common_tensors(reference_root, candidate_root))
    if not paths:
        raise RuntimeError("No common .pt tensors were found")

    failures = 0
    first_failure: Optional[Path] = None
    print(
        "tensor".ljust(48),
        "shape".ljust(22),
        "max_abs".rjust(12),
        "mean_abs".rjust(12),
        "cosine".rjust(12),
        "allclose".rjust(10),
    )
    for relative in paths:
        reference = torch.load(
            reference_root / relative, map_location="cpu", weights_only=True
        )
        candidate = torch.load(
            candidate_root / relative, map_location="cpu", weights_only=True
        )
        if not isinstance(reference, torch.Tensor) or not isinstance(
            candidate, torch.Tensor
        ):
            continue
        reference = trim_target_prefill_padding(
            reference_root, relative, reference
        )
        candidate = trim_target_prefill_padding(
            candidate_root, relative, candidate
        )
        same_shape = tuple(reference.shape) == tuple(candidate.shape)
        if not same_shape:
            failures += 1
            first_failure = first_failure or relative
            print(
                str(relative).ljust(48),
                f"{tuple(reference.shape)} != {tuple(candidate.shape)}",
            )
            continue

        ref = reference.float()
        cand = candidate.float()
        if ref.numel() == 0:
            max_abs = mean_abs = 0.0
            cosine = 1.0
        else:
            diff = ref - cand
            max_abs = float(diff.abs().max().item())
            mean_abs = float(diff.abs().mean().item())
            cosine = float(
                F.cosine_similarity(
                    ref.reshape(1, -1), cand.reshape(1, -1), dim=-1
                ).item()
            )
        close = bool(
            torch.allclose(ref, cand, atol=args.atol, rtol=args.rtol, equal_nan=False)
        )
        if not close:
            failures += 1
            first_failure = first_failure or relative
        print(
            str(relative).ljust(48),
            str(tuple(reference.shape)).ljust(22),
            f"{max_abs:12.5g}",
            f"{mean_abs:12.5g}",
            f"{cosine:12.8f}",
            str(close).rjust(10),
        )
        print_target_hidden_cross_cosine(relative, reference, candidate)

        if relative.name in {
            "base_logits.pt",
            "corrected_logits.pt",
            "target_next_token_logits.pt",
            "target_verify_logits.pt",
            "target_verify_logits_full.pt",
            "bonus_logits.pt",
        } and ref.ndim >= 1:
            k = min(args.topk, ref.shape[-1])
            ref_ids = torch.topk(ref, k=k, dim=-1).indices
            cand_ids = torch.topk(cand, k=k, dim=-1).indices
            top1_equal = float((ref_ids[..., 0] == cand_ids[..., 0]).float().mean())
            overlap = []
            for ref_row, cand_row in zip(
                ref_ids.reshape(-1, k), cand_ids.reshape(-1, k)
            ):
                overlap.append(
                    len(set(ref_row.tolist()) & set(cand_row.tolist())) / float(k)
                )
            print(
                f"  logits: top1_equal={top1_equal:.4f} "
                f"top{k}_overlap={sum(overlap) / len(overlap):.4f}"
            )

    def print_store_checks(label: str, request_root: Path) -> None:
        for expected_path in sorted(
            request_root.glob("kv_inject*/stage*/expected_stored_kv.pt")
        ):
            stored_path = expected_path.with_name("stored_kv.pt")
            if not stored_path.is_file():
                continue
            expected = torch.load(
                expected_path, map_location="cpu", weights_only=True
            ).float()
            stored = torch.load(
                stored_path, map_location="cpu", weights_only=True
            ).float()
            relative = expected_path.relative_to(request_root)
            if tuple(expected.shape) != tuple(stored.shape):
                print(
                    f"  {label} store check {relative.parent}: "
                    f"shape {tuple(expected.shape)} != {tuple(stored.shape)}"
                )
                continue
            diff = expected - stored
            cosine = float(
                F.cosine_similarity(
                    expected.reshape(1, -1), stored.reshape(1, -1), dim=-1
                ).item()
            )
            print(
                f"  {label} store check {relative.parent}: "
                f"max_abs={float(diff.abs().max().item()):.6g}, "
                f"mean_abs={float(diff.abs().mean().item()):.6g}, "
                f"cosine={cosine:.8f}"
            )

    print("\nWithin-device norm/RoPE reference -> stored KV:")
    print_store_checks("reference", reference_root)
    print_store_checks("candidate", candidate_root)

    def load_optional(path: Path):
        if not path.is_file():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def compact_value(value) -> str:
        if not isinstance(value, torch.Tensor):
            return "<missing>"
        return str(value.detach().cpu().tolist())

    def print_round_summaries(label: str, request_root: Path) -> None:
        verify_dirs = sorted(
            path for path in request_root.glob("verify*") if path.is_dir()
        )
        if not verify_dirs:
            print(f"  {label}: no verify captures")
            return
        for path in verify_dirs:
            candidates = load_optional(path / "verify_ids_2d.pt")
            drafts = load_optional(path / "draft_tokens.pt")
            predicts = load_optional(path / "target_predict.pt")
            matches = load_optional(path / "matches.pt")
            correct_len = load_optional(path / "correct_len.pt")
            commit_lens = load_optional(path / "commit_lens.pt")
            bonus = load_optional(path / "bonus.pt")
            out_tokens = load_optional(path / "out_tokens.pt")
            committed = load_optional(path / "committed_token_ids.pt")
            commit_marker = (path / "commit_complete.json").is_file()
            print(f"  {label} {path.name}:")
            print(f"    candidates={compact_value(candidates)}")
            print(f"    drafts={compact_value(drafts)}")
            print(f"    target_predict={compact_value(predicts)}")
            print(f"    matches={compact_value(matches)}")
            print(
                "    accept: "
                f"correct_len={compact_value(correct_len)} "
                f"commit_lens={compact_value(commit_lens)} "
                f"bonus={compact_value(bonus)}"
            )
            print(f"    out_tokens={compact_value(out_tokens)}")
            if committed is not None:
                print(f"    committed_token_ids={compact_value(committed)}")
            print(f"    commit_complete={commit_marker}")

    print("\nSpeculative-round semantic summaries:")
    print_round_summaries("reference", reference_root)
    print_round_summaries("candidate", candidate_root)

    print(f"\nCompared {len(paths)} tensors; non-allclose={failures}.")
    if first_failure is not None:
        print(f"First non-allclose tensor: {first_failure}")
    return 1 if failures and args.fail_on_diff else 0


def _capture(args: argparse.Namespace) -> int:
    dump_dir = Path(args.dump_dir).resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ[_ENV_DUMP_DIR] = str(dump_dir)
    os.environ[_ENV_RID_PREFIX] = args.rid_prefix
    os.environ[_ENV_MAX_CALLS] = str(args.max_calls)
    if args.exit_after_draft:
        os.environ[_ENV_EXIT_AFTER_DRAFT] = "1"
    else:
        os.environ.pop(_ENV_EXIT_AFTER_DRAFT, None)
    if args.exit_after_round:
        os.environ[_ENV_EXIT_AFTER_ROUND] = "1"
    else:
        os.environ.pop(_ENV_EXIT_AFTER_ROUND, None)
    if args.fixed_target_hidden:
        os.environ[_ENV_FIXED_TARGET_HIDDEN] = "1"
        os.environ[_ENV_FIXED_TARGET_HIDDEN_SEED] = str(
            args.fixed_target_hidden_seed
        )
    else:
        os.environ.pop(_ENV_FIXED_TARGET_HIDDEN, None)
        os.environ.pop(_ENV_FIXED_TARGET_HIDDEN_SEED, None)
    _install_capture_patch()

    launch_args = list(args.launch_args)
    if launch_args and launch_args[0] == "--":
        launch_args = launch_args[1:]
    if not launch_args:
        raise ValueError("capture mode requires launch_server arguments after '--'")
    sys.argv = ["sglang.launch_server", *launch_args]
    runpy.run_module("sglang.launch_server", run_name="__main__")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and compare DeepSeek-V4 DSpark's three draft stages."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture", help="launch SGLang with three-stage capture instrumentation"
    )
    capture.add_argument("--dump-dir", required=True)
    capture.add_argument(
        "--rid-prefix",
        default="dspark-stage-compare",
        help="only capture request IDs starting with this prefix",
    )
    capture.add_argument(
        "--max-calls",
        type=int,
        default=8,
        help="maximum draft calls saved per rank and request ID",
    )
    capture.add_argument(
        "--exit-after-draft",
        action="store_true",
        help=(
            "stop each scheduler immediately after saving the first matching "
            "draft proposal, before target verify"
        ),
    )
    capture.add_argument(
        "--exit-after-round",
        action="store_true",
        help=(
            "stop each scheduler after the first complete proposal/verify/"
            "accept/commit round has been saved"
        ),
    )
    capture.add_argument(
        "--fixed-target-hidden",
        action="store_true",
        help=(
            "replace matching requests' injected target hidden states with a "
            "device-independent deterministic BF16 tensor"
        ),
    )
    capture.add_argument(
        "--fixed-target-hidden-seed",
        type=int,
        default=20260723,
        help="integer seed used by --fixed-target-hidden (default: 20260723)",
    )
    capture.add_argument(
        "launch_args",
        nargs=argparse.REMAINDER,
        help="normal sglang.launch_server arguments, placed after '--'",
    )

    compare = subparsers.add_parser(
        "compare", help="compare captures produced on two devices"
    )
    compare.add_argument("--reference", required=True, help="GPU capture root")
    compare.add_argument("--candidate", required=True, help="NPU capture root")
    compare.add_argument("--rank", type=int, default=0)
    compare.add_argument("--rid")
    compare.add_argument("--atol", type=float, default=2e-2)
    compare.add_argument("--rtol", type=float, default=2e-2)
    compare.add_argument("--topk", type=int, default=10)
    compare.add_argument("--fail-on-diff", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "capture":
        return _capture(args)
    if args.command == "compare":
        return _compare(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


# multiprocessing "spawn" imports the parent script as ``__mp_main__``.  The
# environment is inherited, so install the patch during that import as well.
if os.environ.get(_ENV_DUMP_DIR):
    _install_capture_patch()


if __name__ == "__main__":
    raise SystemExit(main())
