#!/usr/bin/env python3
"""Compare target embedding weights from two Safetensors checkpoints.

The default paths compare the original DeepSeek-V4-Flash-DSpark checkpoint
with its W4A8 conversion. Only selected token rows are read by default. Pass
``--full`` to scan the complete embedding matrix in bounded-size chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open


DEFAULT_MODEL_A = "/data/weights/DeepSeek-V4-Flash-DSpark"
DEFAULT_MODEL_B = "/data/weights/DeepSeek-V4-Flash-DSpark-w4a8"
DEFAULT_TOKEN_IDS = "804,9416,470,1148,5409"


@dataclass(frozen=True)
class TensorLocation:
    model_dir: Path
    key: str
    shard: Path


def _iter_safetensor_keys(model_dir: Path) -> Iterable[tuple[str, Path]]:
    index_path = next(
        (
            model_dir / name
            for name in (
                "model.safetensors.index.json",
                "quant_model_weights.safetensors.index.json",
            )
            if (model_dir / name).is_file()
        ),
        None,
    )
    if index_path is None:
        discovered = sorted(model_dir.glob("*.safetensors.index.json"))
        if len(discovered) == 1:
            index_path = discovered[0]
    if index_path is not None:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for key, filename in index["weight_map"].items():
            yield key, model_dir / filename
        return

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"No model.safetensors.index.json or *.safetensors found in {model_dir}"
        )
    for shard in shards:
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    yield key, shard
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read Safetensors shard {shard}. The checkpoint has no "
                "usable model.safetensors.index.json, so every *.safetensors file "
                "must be a complete standard Safetensors file. The shard may be "
                "truncated, contain trailing/non-Safetensors data, or be a leftover "
                "from an interrupted conversion."
            ) from exc


def _embedding_candidates(model_dir: Path) -> list[TensorLocation]:
    candidates = []
    for key, shard in _iter_safetensor_keys(model_dir):
        lower = key.lower()
        if "embed" not in lower or not key.endswith(".weight"):
            continue
        candidates.append(TensorLocation(model_dir, key, shard))
    return candidates


def _candidate_priority(key: str) -> tuple[int, int, str]:
    lower = key.lower()
    is_draft = lower.startswith("mtp.") or ".mtp." in lower
    if key == "model.embed_tokens.weight":
        level = 0
    elif key == "model.model.embed_tokens.weight":
        level = 1
    elif key.endswith(".model.embed_tokens.weight"):
        level = 2
    elif key.endswith(".embed_tokens.weight"):
        level = 3
    elif key.endswith(".embed.weight"):
        level = 4
    else:
        level = 5
    return (1 if is_draft else 0, level, key)


def _resolve_embedding(model_dir: Path, explicit_key: Optional[str]) -> TensorLocation:
    candidates = _embedding_candidates(model_dir)
    if explicit_key is not None:
        matches = [candidate for candidate in candidates if candidate.key == explicit_key]
        if not matches:
            available = "\n  ".join(candidate.key for candidate in candidates)
            raise KeyError(
                f"{explicit_key!r} not found in {model_dir}. "
                f"Embedding-like keys:\n  {available}"
            )
        return matches[0]

    target_candidates = sorted(candidates, key=lambda item: _candidate_priority(item.key))
    if not target_candidates:
        raise KeyError(f"No embedding weight candidate found in {model_dir}")
    return target_candidates[0]


def _parse_token_ids(value: str) -> list[int]:
    token_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not token_ids:
        raise argparse.ArgumentTypeError("at least one token ID is required")
    if min(token_ids) < 0:
        raise argparse.ArgumentTypeError("token IDs must be non-negative")
    return token_ids


def _tensor_digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _print_candidates(label: str, model_dir: Path) -> None:
    print(f"{label} embedding-like checkpoint keys ({model_dir}):")
    for candidate in sorted(
        _embedding_candidates(model_dir), key=lambda item: _candidate_priority(item.key)
    ):
        print(f"  {candidate.key} -> {candidate.shard.name}")


def _compare_selected(
    slice_a,
    slice_b,
    token_ids: list[int],
    vocab_size: int,
    save_dir: Optional[Path],
) -> None:
    invalid = [token_id for token_id in token_ids if token_id >= vocab_size]
    if invalid:
        raise IndexError(f"Token IDs outside vocab size {vocab_size}: {invalid}")

    rows_a = torch.cat(
        [slice_a[token_id : token_id + 1] for token_id in token_ids], dim=0
    ).float()
    rows_b = torch.cat(
        [slice_b[token_id : token_id + 1] for token_id in token_ids], dim=0
    ).float()

    print("\nSelected token rows:")
    print(
        "token_id".ljust(12),
        "cosine".rjust(12),
        "max_abs".rjust(12),
        "mean_abs".rjust(12),
        "a_rms".rjust(12),
        "b_rms".rjust(12),
    )
    for row, token_id in enumerate(token_ids):
        a = rows_a[row]
        b = rows_b[row]
        diff = (a - b).abs()
        cosine = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
        print(
            str(token_id).ljust(12),
            f"{cosine:12.8f}",
            f"{diff.max().item():12.6g}",
            f"{diff.mean().item():12.6g}",
            f"{a.square().mean().sqrt().item():12.6g}",
            f"{b.square().mean().sqrt().item():12.6g}",
        )

    flat_cosine = F.cosine_similarity(
        rows_a.reshape(1, -1), rows_b.reshape(1, -1)
    ).item()
    print(f"\nselected_flat_cosine: {flat_cosine:.10f}")
    print(f"selected_max_abs:     {(rows_a - rows_b).abs().max().item():.10g}")
    print(f"selected_mean_abs:    {(rows_a - rows_b).abs().mean().item():.10g}")
    print(f"selected_a_sha256:    {_tensor_digest(rows_a)}")
    print(f"selected_b_sha256:    {_tensor_digest(rows_b)}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(rows_a.to(torch.bfloat16), save_dir / "embedding_a.pt")
        torch.save(rows_b.to(torch.bfloat16), save_dir / "embedding_b.pt")
        (save_dir / "token_ids.json").write_text(
            json.dumps(token_ids, indent=2), encoding="utf-8"
        )
        print(f"selected rows saved under: {save_dir}")


def _compare_full(slice_a, slice_b, shape: tuple[int, ...], chunk_rows: int) -> None:
    vocab_size = shape[0]
    dot = 0.0
    sum_sq_a = 0.0
    sum_sq_b = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    numel = 0
    top_rows: list[tuple[float, int]] = []

    for start in range(0, vocab_size, chunk_rows):
        end = min(start + chunk_rows, vocab_size)
        a = slice_a[start:end].float()
        b = slice_b[start:end].float()
        diff = (a - b).abs()

        dot += torch.sum(a.double() * b.double()).item()
        sum_sq_a += torch.sum(a.double().square()).item()
        sum_sq_b += torch.sum(b.double().square()).item()
        sum_abs += diff.double().sum().item()
        max_abs = max(max_abs, diff.max().item())
        numel += diff.numel()

        row_mean_abs = diff.flatten(1).mean(dim=1)
        for offset, value in enumerate(row_mean_abs.tolist()):
            item = (float(value), start + offset)
            if len(top_rows) < 10:
                heapq.heappush(top_rows, item)
            elif item[0] > top_rows[0][0]:
                heapq.heapreplace(top_rows, item)

        print(f"\rfull scan: {end}/{vocab_size} rows", end="", flush=True)

    print()
    denominator = math.sqrt(sum_sq_a) * math.sqrt(sum_sq_b)
    cosine = dot / denominator if denominator else float("nan")
    relative_l2 = (
        math.sqrt(max(sum_sq_a + sum_sq_b - 2.0 * dot, 0.0))
        / math.sqrt(sum_sq_a)
        if sum_sq_a
        else float("nan")
    )
    print("\nFull embedding comparison:")
    print(f"cosine:       {cosine:.10f}")
    print(f"max_abs:      {max_abs:.10g}")
    print(f"mean_abs:     {sum_abs / numel:.10g}")
    print(f"relative_l2:  {relative_l2:.10f}")
    print("largest mean-absolute-difference rows:")
    for value, token_id in sorted(top_rows, reverse=True):
        print(f"  token_id={token_id:<8d} mean_abs={value:.10g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    parser.add_argument("--key-a", help="explicit embedding tensor key for model A")
    parser.add_argument("--key-b", help="explicit embedding tensor key for model B")
    parser.add_argument(
        "--token-ids",
        type=_parse_token_ids,
        default=_parse_token_ids(DEFAULT_TOKEN_IDS),
        help=f"comma-separated token IDs (default: {DEFAULT_TOKEN_IDS})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="scan and compare the complete embedding matrix",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
        help="rows loaded per full-scan chunk (default: 256)",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="print embedding-like keys before selecting the target embedding",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="optionally save selected rows from both models as .pt files",
    )
    args = parser.parse_args()

    model_a = Path(args.model_a).resolve()
    model_b = Path(args.model_b).resolve()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    if args.list_keys:
        _print_candidates("A", model_a)
        _print_candidates("B", model_b)

    location_a = _resolve_embedding(model_a, args.key_a)
    location_b = _resolve_embedding(model_b, args.key_b)
    print("Selected target embeddings:")
    print(f"  A: {location_a.key} -> {location_a.shard}")
    print(f"  B: {location_b.key} -> {location_b.shard}")

    with ExitStack() as stack:
        handle_a = stack.enter_context(
            safe_open(location_a.shard, framework="pt", device="cpu")
        )
        handle_b = stack.enter_context(
            safe_open(location_b.shard, framework="pt", device="cpu")
        )
        slice_a = handle_a.get_slice(location_a.key)
        slice_b = handle_b.get_slice(location_b.key)
        shape_a = tuple(slice_a.get_shape())
        shape_b = tuple(slice_b.get_shape())
        # Inspect one row so the chunked mode never materializes the full
        # embedding matrix merely to obtain its dtype.
        dtype_a = slice_a[0:1].dtype
        dtype_b = slice_b[0:1].dtype
        print(f"  A shape/dtype: {shape_a} / {dtype_a}")
        print(f"  B shape/dtype: {shape_b} / {dtype_b}")
        if shape_a != shape_b:
            raise ValueError(f"Embedding shapes differ: {shape_a} != {shape_b}")
        if len(shape_a) != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got {shape_a}")

        _compare_selected(
            slice_a,
            slice_b,
            args.token_ids,
            vocab_size=shape_a[0],
            save_dir=args.save_dir,
        )
        if args.full:
            _compare_full(slice_a, slice_b, shape_a, args.chunk_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
