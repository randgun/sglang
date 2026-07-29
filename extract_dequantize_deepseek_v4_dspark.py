#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract the DSpark ``mtp.*`` weights and dequantize them to BF16.

The source checkpoint is expected to use DeepSeek-V4's mixed format:

* dense DSpark linear weights: FP8 E4M3 + one scale per 128x128 block;
* routed-expert weights: packed MXFP4 + one scale per 32 input values;
* norms, mHC, Markov and confidence parameters: floating-point tensors.

Only tensors whose names start with ``mtp.`` are written. Quantization scales
paired with a converted weight are consumed and omitted. The generated config
has no quantization declaration, so the output can be loaded with
``--speculative-draft-model-quantization unquant``.

Example:

    python3 extract_dequantize_deepseek_v4_dspark.py \
        --source /data/weights/DeepSeek-V4-Flash-DSpark \
        --output /data/weights/DeepSeek-V4-Flash-DSpark-bf16-draft
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SOURCE = "/data/weights/DeepSeek-V4-Flash-DSpark"
DEFAULT_OUTPUT = "/data/weights/DeepSeek-V4-Flash-DSpark-bf16-draft"
DSPARK_PREFIX = "mtp."

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
_BYTE_VALUES = torch.arange(256, dtype=torch.uint8)
FP4_PAIR_TABLE = torch.stack(
    (
        FP4_TABLE[(_BYTE_VALUES & 0x0F).long()],
        FP4_TABLE[(_BYTE_VALUES >> 4).long()],
    ),
    dim=-1,
)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def paired_scale_name(weight_name: str) -> str:
    return f"{weight_name.removesuffix('.weight')}.scale"


def float8_weight_dtypes() -> set[torch.dtype]:
    result: set[torch.dtype] = set()
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            result.add(dtype)
    return result


def packed_fp4_dtypes() -> set[torch.dtype]:
    result = {torch.int8, torch.uint8}
    dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if dtype is not None:
        result.add(dtype)
    return result


def dequantize_expert_fp4(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Expand packed E2M1 values and apply per-32 MXFP scales."""
    if packed_weight.dtype not in packed_fp4_dtypes():
        raise ValueError(
            f"expected packed FP4 weight, got dtype={packed_weight.dtype}"
        )
    if packed_weight.ndim != 2:
        raise ValueError(
            f"expected a 2D packed FP4 weight, got {packed_weight.ndim}D"
        )

    out_dim, packed_in_dim = packed_weight.shape
    in_dim = packed_in_dim * 2
    if in_dim % 32:
        raise ValueError(f"FP4 input dimension must be divisible by 32, got {in_dim}")

    expected_scale_shape = (out_dim, in_dim // 32)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            "unexpected FP4 scale shape: "
            f"weight={tuple(packed_weight.shape)}, "
            f"scale={tuple(weight_scale.shape)}, "
            f"expected={expected_scale_shape}"
        )

    packed_u8 = packed_weight.view(torch.uint8)
    values = FP4_PAIR_TABLE[packed_u8.long()].reshape(out_dim, in_dim)
    scale = weight_scale.float().repeat_interleave(32, dim=1)
    return (values * scale).to(torch.bfloat16).contiguous()


def dequantize_dense_fp8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """Apply a 2D block scale to a serialized FP8 dense weight."""
    if weight.dtype not in float8_weight_dtypes():
        raise ValueError(f"expected an FP8 weight, got dtype={weight.dtype}")
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D FP8 weight, got {weight.ndim}D")

    block_n, block_k = block_size
    n, k = weight.shape
    expected_scale_shape = (
        (n + block_n - 1) // block_n,
        (k + block_k - 1) // block_k,
    )
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            "unexpected FP8 scale shape: "
            f"weight={tuple(weight.shape)}, "
            f"scale={tuple(weight_scale.shape)}, "
            f"block_size={block_size}, "
            f"expected={expected_scale_shape}"
        )

    scale = weight_scale.float().repeat_interleave(block_n, dim=0)
    scale = scale.repeat_interleave(block_k, dim=1)[:n, :k]
    return (weight.float() * scale).to(torch.bfloat16).contiguous()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def get_block_size(source: Path, source_index: dict[str, Any]) -> tuple[int, int]:
    raw = source_index.get("metadata", {}).get("weight_block_size")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None

    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        config = read_json(source / "config.json")
        quant_config = config.get("quantization_config") or {}
        raw = quant_config.get("weight_block_size", [128, 128])

    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"invalid FP8 weight_block_size: {raw!r}")
    block_size = int(raw[0]), int(raw[1])
    if min(block_size) <= 0:
        raise ValueError(f"FP8 weight_block_size must be positive: {block_size}")
    return block_size


def copy_sidecars(source: Path, output: Path) -> None:
    skipped = {
        "config.json",
        "model.safetensors.index.json",
        "quant_model_description.json",
        "hf_quant_config.json",
        "conversion_report.json",
    }
    for item in source.iterdir():
        if item.name.startswith(".") or item.name in skipped:
            continue
        if item.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
            continue

        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, destination)


def write_draft_config(source: Path, output: Path) -> None:
    config = read_json(source / "config.json")
    required = (
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    )
    missing = [name for name in required if config.get(name) is None]
    if missing:
        raise ValueError(f"source config is missing DSpark fields: {missing}")

    config.pop("quantization_config", None)
    config.pop("expert_dtype", None)
    config["torch_dtype"] = "bfloat16"
    config["dspark_draft_only"] = True
    config["dspark_mtp_dequantized_to_bf16"] = True
    write_json(output / "config.json", config)


class TensorReader:
    """Read indexed tensors without requiring weight/scale shard co-location."""

    def __init__(self, source: Path, weight_map: dict[str, str]) -> None:
        self.source = source
        self.weight_map = weight_map

    def get(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map[name]
        with safe_open(
            self.source / shard_name, framework="pt", device="cpu"
        ) as shard:
            return shard.get_tensor(name)


def save_current_shard(
    output: Path,
    tensors: OrderedDict[str, torch.Tensor],
    shard_index: int,
    output_weight_map: dict[str, str],
) -> tuple[int, int]:
    if not tensors:
        return shard_index, 0

    filename = f"model-{shard_index:05d}.safetensors"
    save_file(dict(tensors), output / filename, metadata={"format": "pt"})
    written_bytes = sum(tensor_nbytes(tensor) for tensor in tensors.values())
    for name in tensors:
        output_weight_map[name] = filename
    tensors.clear()
    return shard_index + 1, written_bytes


def rename_output_shards(
    output: Path,
    output_weight_map: dict[str, str],
    num_shards: int,
) -> dict[str, str]:
    filename_map: dict[str, str] = {}
    for shard_index in range(1, num_shards + 1):
        old_name = f"model-{shard_index:05d}.safetensors"
        new_name = f"model-{shard_index:05d}-of-{num_shards:05d}.safetensors"
        (output / old_name).rename(output / new_name)
        filename_map[old_name] = new_name
    return {name: filename_map[filename] for name, filename in output_weight_map.items()}


def classify_and_convert(
    name: str,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, str]:
    if scale is None:
        return weight.contiguous(), f"preserve_{str(weight.dtype).removeprefix('torch.')}"

    if ".experts." in name and weight.dtype in packed_fp4_dtypes():
        return dequantize_expert_fp4(weight, scale), "mxfp4_to_bf16"
    if weight.dtype in float8_weight_dtypes():
        return dequantize_dense_fp8(weight, scale, block_size), "fp8_to_bf16"

    raise ValueError(
        f"paired DSpark tensor {name!r} has unsupported dtype={weight.dtype}; "
        f"scale dtype={scale.dtype}, weight shape={tuple(weight.shape)}, "
        f"scale shape={tuple(scale.shape)}"
    )


def convert(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output or source in output.parents:
        raise ValueError("--output must not equal --source or be inside it")
    if args.max_shard_gb <= 0:
        raise ValueError("--max-shard-gb must be positive")

    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing source index: {index_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"missing source config: {config_path}")

    source_index = read_json(index_path)
    source_weight_map: dict[str, str] = source_index.get("weight_map") or {}
    selected_names = sorted(
        name for name in source_weight_map if name.startswith(DSPARK_PREFIX)
    )
    if not selected_names:
        raise ValueError(f"no {DSPARK_PREFIX!r} tensors found in {index_path}")

    paired_weights = {
        name
        for name in selected_names
        if name.endswith(".weight")
        and paired_scale_name(name) in source_weight_map
        and paired_scale_name(name).startswith(DSPARK_PREFIX)
    }
    consumed_scales = {paired_scale_name(name) for name in paired_weights}
    output_names = [name for name in selected_names if name not in consumed_scales]

    if args.list_only:
        summary = {
            "source": str(source),
            "selected_mtp_tensors": len(selected_names),
            "output_tensors": len(output_names),
            "paired_quant_weights": len(paired_weights),
            "consumed_quant_scales": len(consumed_scales),
            "sample_output_names": output_names[:50],
        }
        print(json.dumps(summary, indent=2), flush=True)
        return

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copy_sidecars(source, output)
    write_draft_config(source, output)

    block_size = get_block_size(source, source_index)
    reader = TensorReader(source, source_weight_map)
    max_shard_bytes = int(args.max_shard_gb * 1024**3)
    output_weight_map: dict[str, str] = {}
    pending: OrderedDict[str, torch.Tensor] = OrderedDict()
    pending_bytes = 0
    shard_index = 1
    total_tensor_bytes = 0
    counters: Counter[str] = Counter()
    start = time.time()

    # Preserve source-shard order to avoid random I/O. Cross-shard scale pairs
    # are handled by TensorReader and do not require loading all source shards.
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name in output_names:
        names_by_shard[source_weight_map[name]].append(name)
    source_shards = list(OrderedDict.fromkeys(source_weight_map.values()))

    processed = 0
    for source_shard_index, shard_name in enumerate(source_shards, start=1):
        names = names_by_shard.get(shard_name, [])
        if not names:
            continue
        shard_start = time.time()
        converted_in_shard = 0
        with safe_open(source / shard_name, framework="pt", device="cpu") as shard:
            for name in names:
                if args.max_tensors and processed >= args.max_tensors:
                    break

                weight = shard.get_tensor(name)
                scale_name = paired_scale_name(name) if name in paired_weights else None
                if scale_name is None:
                    scale = None
                elif source_weight_map[scale_name] == shard_name:
                    scale = shard.get_tensor(scale_name)
                else:
                    scale = reader.get(scale_name)
                    counters["cross_shard_pairs"] += 1

                output_tensor, action = classify_and_convert(
                    name=name,
                    weight=weight,
                    scale=scale,
                    block_size=block_size,
                )
                counters[action] += 1

                nbytes = tensor_nbytes(output_tensor)
                if pending and pending_bytes + nbytes > max_shard_bytes:
                    shard_index, _ = save_current_shard(
                        output, pending, shard_index, output_weight_map
                    )
                    pending_bytes = 0

                pending[name] = output_tensor
                pending_bytes += nbytes
                total_tensor_bytes += nbytes
                processed += 1
                converted_in_shard += 1

        print(
            f"[{source_shard_index}/{len(source_shards)}] {shard_name}: "
            f"output_tensors={converted_in_shard}, total={processed}, "
            f"elapsed={time.time() - shard_start:.1f}s",
            flush=True,
        )
        if args.max_tensors and processed >= args.max_tensors:
            break

    shard_index, _ = save_current_shard(
        output, pending, shard_index, output_weight_map
    )
    num_output_shards = shard_index - 1
    if not num_output_shards:
        raise RuntimeError("conversion produced no output shards")

    renamed_weight_map = rename_output_shards(
        output, output_weight_map, num_output_shards
    )
    index = {
        "metadata": {
            "format": "dspark_mtp_bf16",
            "source_model": source.name,
            "total_size": total_tensor_bytes,
            "source_mtp_tensors": len(selected_names),
            "consumed_quant_scales": len(consumed_scales),
        },
        "weight_map": renamed_weight_map,
    }
    write_json(output / "model.safetensors.index.json", index)

    report = {
        "source": str(source),
        "output": str(output),
        "elapsed_seconds": round(time.time() - start, 3),
        "source_mtp_tensors": len(selected_names),
        "output_tensors": len(renamed_weight_map),
        "output_shards": num_output_shards,
        "output_tensor_bytes": total_tensor_bytes,
        "output_file_bytes": sum(
            path.stat().st_size for path in output.glob("model-*.safetensors")
        ),
        "actions": dict(counters),
        "partial_debug_output": bool(args.max_tensors),
    }
    write_json(output / "conversion_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Source checkpoint directory (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output BF16 draft directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=4.0,
        help="Maximum output shard tensor payload in GiB (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before conversion.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Inspect mtp tensors and quantized pairs without writing output.",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="Write at most N output tensors; for converter debugging only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
