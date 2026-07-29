#!/usr/bin/env python3
"""Smoke-test the standalone vLLM-Ascend DSpark sparse-attention operators.

This script intentionally does not import vllm or vllm_ascend. It loads the
extension with torch.ops.load_library and directly invokes:

  torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
  torch.ops._C_ascend.npu_sparse_attn_sharedkv

Example:

  python3 scripts/test_dspark_sparse_attn_raw.py \
    --so /home/kelon/code/vllm-ascend/build/\
vllm_ascend_C.cpython-311-aarch64-linux-gnu.so \
    --mode both
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # Register the NPU/PrivateUse1 dispatch.


METADATA_OP = "npu_sparse_attn_sharedkv_metadata"
ATTENTION_OP = "npu_sparse_attn_sharedkv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--so",
        default=os.getenv("SGLANG_DSPARK_VLLM_ASCEND_SO"),
        help=(
            "Absolute path to vllm_ascend_C*.so. Defaults to "
            "SGLANG_DSPARK_VLLM_ASCEND_SO."
        ),
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--mode",
        choices=("basic", "dspark", "both"),
        default="both",
        help="Test causal SWA, DSpark block-noncausal SWA, or both.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-len", type=int, default=5)
    parser.add_argument("--kv-len", type=int, default=5)
    parser.add_argument("--num-heads-q", type=int, default=64)
    parser.add_argument("--num-heads-kv", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="Optionally save inputs, metadata, outputs, and summary here.",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the registered operator schemas.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_operators(so_path: str | None, print_schema: bool) -> Path:
    if not so_path:
        fail(
            "Missing --so. Pass the extension path explicitly or set "
            "SGLANG_DSPARK_VLLM_ASCEND_SO."
        )

    path = Path(so_path).expanduser().resolve()
    if not path.is_file():
        fail(f"Extension does not exist: {path}")
    if f"cpython-{sys.version_info.major}{sys.version_info.minor}" not in path.name:
        fail(
            f"Python ABI mismatch: running Python {sys.version_info.major}."
            f"{sys.version_info.minor}, but extension is {path.name}"
        )

    print(f"[load] {path}", flush=True)
    torch.ops.load_library(str(path))

    namespace = torch.ops._C_ascend
    for name in (METADATA_OP, ATTENTION_OP):
        if not hasattr(namespace, name):
            fail(f"Extension did not register torch.ops._C_ascend.{name}")
        packet = getattr(namespace, name)
        print(f"[registered] torch.ops._C_ascend.{name}", flush=True)
        if print_schema:
            for overload_name in packet.overloads():
                overload = getattr(packet, overload_name)
                print(f"  {overload._schema}", flush=True)
    return path


def synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def cumulative_lengths(batch_size: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        0,
        (batch_size + 1) * length,
        length,
        dtype=torch.int32,
        device=device,
    )


def make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if args.batch_size <= 0:
        fail("--batch-size must be positive")
    if args.query_len <= 0 or args.kv_len <= 0:
        fail("--query-len and --kv-len must be positive")
    if args.query_len > args.kv_len:
        fail("This smoke test requires --query-len <= --kv-len")
    if args.page_size <= 0:
        fail("--page-size must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    total_q = args.batch_size * args.query_len
    pages_per_request = math.ceil(args.kv_len / args.page_size)
    num_pages = args.batch_size * pages_per_request

    # TND: [total query tokens, query heads, head dimension].
    q = torch.randn(
        total_q,
        args.num_heads_q,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    # PA_ND: [physical pages, page size, KV heads, head dimension].
    ori_kv = torch.randn(
        num_pages,
        args.page_size,
        args.num_heads_kv,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    block_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(args.batch_size, pages_per_request)
    cu_seqlens_q = cumulative_lengths(args.batch_size, args.query_len, device)
    seqused_kv = torch.full(
        (args.batch_size,),
        args.kv_len,
        dtype=torch.int32,
        device=device,
    )
    return {
        "q": q,
        "ori_kv": ori_kv,
        "ori_block_table": block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_kv": seqused_kv,
    }


def make_dspark_sparse_indices(
    args: argparse.Namespace,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Build [T, N_kv, K] physical slot IDs for one whole draft block."""
    index_width = math.ceil(
        (args.window_size + args.query_len) / 128
    ) * 128
    if index_width > 2048:
        fail(f"DSpark sparse index width exceeds kernel limit: {index_width}")

    rows = []
    for request_id in range(args.batch_size):
        start_pos = max(0, args.kv_len - args.window_size - args.query_len)
        positions = torch.arange(
            start_pos,
            args.kv_len,
            dtype=torch.int64,
            device=block_table.device,
        )
        page_columns = positions // args.page_size
        offsets = positions % args.page_size
        page_ids = block_table[request_id].index_select(0, page_columns)
        slots = page_ids.to(torch.int64) * args.page_size + offsets
        padded = torch.full(
            (index_width,), -1, dtype=torch.int32, device=block_table.device
        )
        padded[: slots.numel()] = slots.to(torch.int32)
        rows.extend([padded] * args.query_len)

    return (
        torch.stack(rows, dim=0)
        .unsqueeze(1)
        .expand(-1, args.num_heads_kv, -1)
        .contiguous()
    )


def tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "rms": float(value.square().mean().sqrt().item()),
    }


def run_case(
    args: argparse.Namespace,
    case_name: str,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    dspark = case_name == "dspark"
    ori_win_left = (
        args.window_size + args.query_len - 1 if dspark else args.window_size - 1
    )
    sparse_indices = (
        make_dspark_sparse_indices(args, inputs["ori_block_table"])
        if dspark
        else None
    )

    metadata_kwargs = {
        "num_heads_q": args.num_heads_q,
        "num_heads_kv": args.num_heads_kv,
        "head_dim": args.head_dim,
        "cu_seqlens_q": inputs["cu_seqlens_q"],
        # This matches the current vLLM/SGLang DSpark draft metadata path.
        "cu_seqlens_ori_kv": inputs["cu_seqlens_q"],
        "seqused_kv": inputs["seqused_kv"],
        "batch_size": args.batch_size,
        "max_seqlen_q": args.query_len,
        "max_seqlen_kv": args.kv_len,
        "cmp_ratio": 1,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": ori_win_left,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "PA_ND",
        "has_ori_kv": True,
        "has_cmp_kv": False,
        "device": args.device,
    }
    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        **metadata_kwargs
    )
    synchronize(torch.device(args.device))

    attention_kwargs = {
        "ori_kv": inputs["ori_kv"],
        "ori_block_table": inputs["ori_block_table"],
        "cu_seqlens_q": inputs["cu_seqlens_q"],
        "cu_seqlens_ori_kv": inputs["cu_seqlens_q"],
        "seqused_kv": inputs["seqused_kv"],
        "metadata": metadata,
        "softmax_scale": args.head_dim**-0.5,
        "cmp_ratio": 1,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": ori_win_left,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "PA_ND",
        "return_softmax_lse": True,
    }
    if sparse_indices is not None:
        attention_kwargs["ori_sparse_indices"] = sparse_indices

    def execute() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            inputs["q"], **attention_kwargs
        )

    for _ in range(args.warmup):
        execute()
    synchronize(torch.device(args.device))

    begin = time.perf_counter()
    outputs = [execute() for _ in range(args.repeat)]
    synchronize(torch.device(args.device))
    elapsed_ms = (time.perf_counter() - begin) * 1000 / args.repeat

    out, softmax_lse = outputs[-1]
    out_stats = tensor_stats(out)
    if not out_stats["finite"]:
        fail(f"{case_name}: attention output contains NaN or Inf")
    if tuple(out.shape) != tuple(inputs["q"].shape):
        fail(
            f"{case_name}: output shape {tuple(out.shape)} does not match "
            f"query shape {tuple(inputs['q'].shape)}"
        )

    # Repeated calls over immutable inputs should be deterministic.
    max_repeat_diff = 0.0
    for previous_out, _ in outputs[:-1]:
        diff = (previous_out.float() - out.float()).abs().max().item()
        max_repeat_diff = max(max_repeat_diff, float(diff))

    result = {
        "case": case_name,
        "metadata": tensor_stats(metadata),
        "output": out_stats,
        "softmax_lse": tensor_stats(softmax_lse),
        "ori_win_left": ori_win_left,
        "sparse_indices_shape": (
            list(sparse_indices.shape) if sparse_indices is not None else None
        ),
        "average_ms": elapsed_ms,
        "max_repeat_diff": max_repeat_diff,
    }
    tensors = {
        **inputs,
        "metadata": metadata,
        "output": out,
        "softmax_lse": softmax_lse,
    }
    if sparse_indices is not None:
        tensors["ori_sparse_indices"] = sparse_indices
    return result, tensors


def dump_case(
    dump_dir: Path,
    case_name: str,
    result: dict[str, object],
    tensors: dict[str, torch.Tensor],
) -> None:
    case_dir = dump_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in tensors.items():
        torch.save(tensor.detach().cpu(), case_dir / f"{name}.pt")
    (case_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> int:
    args = parse_args()
    so_path = load_operators(args.so, args.print_schema)
    if not torch.npu.is_available():
        fail("torch.npu.is_available() is False")
    if args.repeat <= 0:
        fail("--repeat must be positive")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    inputs = make_inputs(args)
    cases = ("basic", "dspark") if args.mode == "both" else (args.mode,)

    summary = {
        "extension": str(so_path),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": args.device,
        "cases": [],
    }
    for case_name in cases:
        print(f"\n[run] case={case_name}", flush=True)
        result, tensors = run_case(args, case_name, inputs)
        summary["cases"].append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if args.dump_dir is not None:
            dump_case(args.dump_dir, case_name, result, tensors)

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        (args.dump_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"\n[dump] {args.dump_dir.resolve()}", flush=True)

    print("\nPASS: metadata and attention operators executed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
