#!/usr/bin/env python3
"""Compare the target LM head and DSpark stage-2 head in one checkpoint."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


DEFAULT_MODEL = "/data/weights/DeepSeek-V4-Flash-DSpark-w4a8"
DEFAULT_TARGET_KEY = "head.weight"
DEFAULT_DSPARK_KEY = "mtp.2.head.weight"
DEFAULT_TOKEN_IDS = "0,1,804,9416,470,1148,5409,128799"


@dataclass(frozen=True)
class TensorLocation:
    key: str
    shard: Path


class EffectiveTensorSlice:
    """Safetensors row reader with optional block-FP8 dequantization."""

    def __init__(
        self,
        weight_slice,
        *,
        scale_slice=None,
        block_size: tuple[int, int] = (128, 128),
    ):
        self.weight_slice = weight_slice
        self.scale_slice = scale_slice
        self.block_n, self.block_k = block_size
        self.shape = tuple(weight_slice.get_shape())

    def get_shape(self):
        return self.shape

    def __getitem__(self, row_slice):
        weight = self.weight_slice[row_slice]
        if self.scale_slice is None:
            return weight
        if not isinstance(row_slice, slice):
            raise TypeError("block-FP8 reader requires a contiguous row slice")
        start = 0 if row_slice.start is None else int(row_slice.start)
        stop = self.shape[0] if row_slice.stop is None else int(row_slice.stop)
        if row_slice.step not in (None, 1):
            raise ValueError("block-FP8 reader does not support strided slices")
        first_scale_row = start // self.block_n
        last_scale_row = (stop - 1) // self.block_n
        scale_blocks = self.scale_slice[
            first_scale_row : last_scale_row + 1
        ].float()
        scale_row_ids = (
            torch.arange(start, stop, dtype=torch.int64) // self.block_n
            - first_scale_row
        )
        row_scales = scale_blocks[scale_row_ids]
        expanded_scale = row_scales.repeat_interleave(self.block_k, dim=1)
        expanded_scale = expanded_scale[:, : self.shape[1]]
        return weight.float() * expanded_scale


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    preferred_names = (
        "model.safetensors.index.json",
        "quant_model_weights.safetensors.index.json",
    )
    index_path = next(
        (
            model_dir / name
            for name in preferred_names
            if (model_dir / name).is_file()
        ),
        None,
    )
    if index_path is None:
        discovered = sorted(model_dir.glob("*.safetensors.index.json"))
        if len(discovered) == 1:
            index_path = discovered[0]
        elif len(discovered) > 1:
            raise FileNotFoundError(
                f"Multiple Safetensors indexes found in {model_dir}: "
                f"{[path.name for path in discovered]}. Rename one to a supported "
                "standard name or remove stale indexes."
            )
    if index_path is None:
        raise FileNotFoundError(
            f"No *.safetensors.index.json found in {model_dir}. "
            "This script requires an indexed checkpoint."
        )
    print(f"Using checkpoint index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} has no valid weight_map")
    return weight_map


def _block_size(model_dir: Path) -> tuple[int, int]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return (128, 128)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config") or {}
    value = quant_config.get("weight_block_size", [128, 128])
    if not isinstance(value, list) or len(value) != 2:
        return (128, 128)
    return int(value[0]), int(value[1])


def _paired_scale_location(
    model_dir: Path,
    weight_map: dict[str, str],
    weight_key: str,
) -> TensorLocation | None:
    if not weight_key.endswith(".weight"):
        return None
    scale_key = weight_key.removesuffix(".weight") + ".scale"
    if scale_key not in weight_map:
        return None
    return TensorLocation(scale_key, model_dir / weight_map[scale_key])


def _resolve(
    model_dir: Path, weight_map: dict[str, str], requested_key: str
) -> TensorLocation:
    if requested_key in weight_map:
        return TensorLocation(requested_key, model_dir / weight_map[requested_key])

    suffix_matches = sorted(
        key for key in weight_map if key.endswith(f".{requested_key}")
    )
    if len(suffix_matches) == 1:
        key = suffix_matches[0]
        return TensorLocation(key, model_dir / weight_map[key])

    related = sorted(
        key
        for key in weight_map
        if "head" in key.lower() and key.endswith(".weight")
    )
    formatted = "\n  ".join(related)
    raise KeyError(
        f"Could not uniquely resolve {requested_key!r}. "
        f"Head-like weight keys:\n  {formatted}"
    )


def _parse_token_ids(value: str) -> list[int]:
    token_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not token_ids:
        raise argparse.ArgumentTypeError("at least one token ID is required")
    if min(token_ids) < 0:
        raise argparse.ArgumentTypeError("token IDs must be non-negative")
    return token_ids


def _compare_selected(
    target_slice,
    dspark_slice,
    token_ids: list[int],
    vocab_size: int,
) -> None:
    invalid = [token_id for token_id in token_ids if token_id >= vocab_size]
    if invalid:
        raise IndexError(f"Token IDs outside vocab size {vocab_size}: {invalid}")

    print("\nSelected vocabulary rows:")
    print(
        "token_id".ljust(12),
        "cosine".rjust(12),
        "max_abs".rjust(12),
        "mean_abs".rjust(12),
        "target_rms".rjust(12),
        "dspark_rms".rjust(12),
    )
    for token_id in token_ids:
        target = target_slice[token_id : token_id + 1].float().flatten()
        dspark = dspark_slice[token_id : token_id + 1].float().flatten()
        diff = (target - dspark).abs()
        cosine = F.cosine_similarity(
            target.reshape(1, -1), dspark.reshape(1, -1)
        ).item()
        print(
            str(token_id).ljust(12),
            f"{cosine:12.8f}",
            f"{diff.max().item():12.6g}",
            f"{diff.mean().item():12.6g}",
            f"{target.square().mean().sqrt().item():12.6g}",
            f"{dspark.square().mean().sqrt().item():12.6g}",
        )


def _compare_full(
    target_slice,
    dspark_slice,
    shape: tuple[int, int],
    chunk_rows: int,
    atol: float,
    rtol: float,
) -> None:
    vocab_size, _ = shape
    dot = 0.0
    sum_sq_target = 0.0
    sum_sq_dspark = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    numel = 0
    close_numel = 0
    exact_numel = 0
    row_cosine_sum = 0.0
    row_cosine_min = float("inf")
    row_cosine_count = 0
    top_rows: list[tuple[float, int, float]] = []

    for start in range(0, vocab_size, chunk_rows):
        end = min(start + chunk_rows, vocab_size)
        target = target_slice[start:end].float()
        dspark = dspark_slice[start:end].float()
        diff = (target - dspark).abs()

        target64 = target.double()
        dspark64 = dspark.double()
        dot += torch.sum(target64 * dspark64).item()
        sum_sq_target += torch.sum(target64.square()).item()
        sum_sq_dspark += torch.sum(dspark64.square()).item()
        sum_abs += diff.double().sum().item()
        max_abs = max(max_abs, diff.max().item())
        numel += diff.numel()
        close_numel += torch.isclose(
            target, dspark, atol=atol, rtol=rtol
        ).sum().item()
        exact_numel += torch.eq(target, dspark).sum().item()

        row_cosines = F.cosine_similarity(target, dspark, dim=-1)
        row_cosine_sum += row_cosines.double().sum().item()
        row_cosine_min = min(row_cosine_min, row_cosines.min().item())
        row_cosine_count += row_cosines.numel()
        row_mean_abs = diff.mean(dim=-1)
        for offset, (mean_abs, row_cosine) in enumerate(
            zip(row_mean_abs.tolist(), row_cosines.tolist())
        ):
            item = (float(mean_abs), start + offset, float(row_cosine))
            if len(top_rows) < 10:
                heapq.heappush(top_rows, item)
            elif item[0] > top_rows[0][0]:
                heapq.heapreplace(top_rows, item)

        print(f"\rfull scan: {end}/{vocab_size} rows", end="", flush=True)

    print()
    denominator = math.sqrt(sum_sq_target) * math.sqrt(sum_sq_dspark)
    global_cosine = dot / denominator if denominator else float("nan")
    diff_sq = max(sum_sq_target + sum_sq_dspark - 2.0 * dot, 0.0)
    relative_l2 = (
        math.sqrt(diff_sq) / math.sqrt(sum_sq_target)
        if sum_sq_target
        else float("nan")
    )
    # Best scalar alpha for approximating DSpark head as alpha * target head.
    alpha = dot / sum_sq_target if sum_sq_target else float("nan")
    scaled_residual_sq = max(
        sum_sq_dspark - 2.0 * alpha * dot + alpha * alpha * sum_sq_target,
        0.0,
    )
    scaled_relative_l2 = (
        math.sqrt(scaled_residual_sq) / math.sqrt(sum_sq_dspark)
        if sum_sq_dspark
        else float("nan")
    )

    print("\nFull head comparison:")
    print(f"global_cosine:          {global_cosine:.10f}")
    print(f"mean_row_cosine:        {row_cosine_sum / row_cosine_count:.10f}")
    print(f"minimum_row_cosine:     {row_cosine_min:.10f}")
    print(f"max_abs:                {max_abs:.10g}")
    print(f"mean_abs:               {sum_abs / numel:.10g}")
    print(f"relative_l2:            {relative_l2:.10f}")
    print(f"exact_fraction:         {exact_numel / numel:.10f}")
    print(f"allclose_fraction:      {close_numel / numel:.10f}")
    print(f"allclose_atol/rtol:     {atol} / {rtol}")
    print(f"best_scalar_alpha:      {alpha:.10f}")
    print(f"scaled_relative_l2:     {scaled_relative_l2:.10f}")
    print("largest mean-absolute-difference rows:")
    for mean_abs, token_id, row_cosine in sorted(top_rows, reverse=True):
        print(
            f"  token_id={token_id:<8d} mean_abs={mean_abs:.10g} "
            f"cosine={row_cosine:.8f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--target-model",
        help="checkpoint containing --target-key (default: --model)",
    )
    parser.add_argument(
        "--dspark-model",
        help="checkpoint containing --dspark-key (default: --model)",
    )
    parser.add_argument("--target-key", default=DEFAULT_TARGET_KEY)
    parser.add_argument("--dspark-key", default=DEFAULT_DSPARK_KEY)
    parser.add_argument(
        "--token-ids",
        type=_parse_token_ids,
        default=_parse_token_ids(DEFAULT_TOKEN_IDS),
        help=f"comma-separated vocabulary rows (default: {DEFAULT_TOKEN_IDS})",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
        help="rows loaded per chunk (default: 256)",
    )
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="compare selected token rows without scanning the full heads",
    )
    args = parser.parse_args()

    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    target_model_dir = Path(args.target_model or args.model).expanduser().resolve()
    dspark_model_dir = Path(args.dspark_model or args.model).expanduser().resolve()
    target_weight_map = _load_weight_map(target_model_dir)
    if dspark_model_dir == target_model_dir:
        dspark_weight_map = target_weight_map
    else:
        dspark_weight_map = _load_weight_map(dspark_model_dir)
    target_location = _resolve(
        target_model_dir, target_weight_map, args.target_key
    )
    dspark_location = _resolve(
        dspark_model_dir, dspark_weight_map, args.dspark_key
    )

    print("Resolved tensors:")
    print(
        f"  target: {target_location.key} -> {target_location.shard}"
    )
    print(
        f"  dspark: {dspark_location.key} -> {dspark_location.shard}"
    )

    with ExitStack() as stack:
        handles = {}

        def get_handle(shard: Path):
            if shard not in handles:
                handles[shard] = stack.enter_context(
                    safe_open(shard, framework="pt", device="cpu")
                )
            return handles[shard]

        target_raw_slice = get_handle(target_location.shard).get_slice(
            target_location.key
        )
        dspark_raw_slice = get_handle(dspark_location.shard).get_slice(
            dspark_location.key
        )
        target_raw_sample = target_raw_slice[0:1]
        dspark_raw_sample = dspark_raw_slice[0:1]
        target_scale_location = _paired_scale_location(
            target_model_dir, target_weight_map, target_location.key
        )
        dspark_scale_location = _paired_scale_location(
            dspark_model_dir, dspark_weight_map, dspark_location.key
        )
        target_uses_scale = (
            target_scale_location is not None
            and "float8" in str(target_raw_sample.dtype)
        )
        dspark_uses_scale = (
            dspark_scale_location is not None
            and "float8" in str(dspark_raw_sample.dtype)
        )
        target_scale_slice = (
            get_handle(target_scale_location.shard).get_slice(
                target_scale_location.key
            )
            if target_uses_scale
            else None
        )
        dspark_scale_slice = (
            get_handle(dspark_scale_location.shard).get_slice(
                dspark_scale_location.key
            )
            if dspark_uses_scale
            else None
        )
        target_slice = EffectiveTensorSlice(
            target_raw_slice,
            scale_slice=target_scale_slice,
            block_size=_block_size(target_model_dir),
        )
        dspark_slice = EffectiveTensorSlice(
            dspark_raw_slice,
            scale_slice=dspark_scale_slice,
            block_size=_block_size(dspark_model_dir),
        )
        target_shape = tuple(target_slice.get_shape())
        dspark_shape = tuple(dspark_slice.get_shape())
        target_sample = target_slice[0:1]
        dspark_sample = dspark_slice[0:1]
        target_dtype = target_raw_sample.dtype
        dspark_dtype = dspark_raw_sample.dtype

        print(
            f"  target shape/raw dtype: {target_shape} / {target_dtype}"
            + (
                f" (dequantized with {target_scale_location.key})"
                if target_uses_scale
                else ""
            )
        )
        print(
            f"  dspark shape/raw dtype: {dspark_shape} / {dspark_dtype}"
            + (
                f" (dequantized with {dspark_scale_location.key})"
                if dspark_uses_scale
                else ""
            )
        )
        if target_shape != dspark_shape:
            raise ValueError(f"Head shapes differ: {target_shape} != {dspark_shape}")
        if len(target_shape) != 2:
            raise ValueError(f"Expected 2D head weights, got {target_shape}")
        if not target_sample.is_floating_point() or not dspark_sample.is_floating_point():
            raise TypeError(
                "At least one head is stored in a packed/integer quantized form. "
                "Raw integer tensors are not numerically comparable; dequantize "
                "the corresponding weight using its ModelSlim scale first."
            )

        _compare_selected(
            target_slice,
            dspark_slice,
            args.token_ids,
            vocab_size=target_shape[0],
        )
        if not args.selected_only:
            _compare_full(
                target_slice,
                dspark_slice,
                target_shape,
                args.chunk_rows,
                args.atol,
                args.rtol,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
