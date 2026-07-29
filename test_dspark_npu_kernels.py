#!/usr/bin/env python3
"""Smoke-test DSpark kernel dispatch and Torch fallbacks on Ascend NPU.

Run from the SGLang repository root:

    python3 test_dspark_npu_kernels.py --device npu:0

Each case runs in a fresh subprocess by default.  This is intentional: a bad
Triton/CUDA dispatch can poison the accelerator context and hide later results.
The test compares each public ``execute`` path against its ``torch`` reference
implementation and reports which paths still need an NPU implementation or a
dispatch fix.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parent
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


CASE_NAMES = [
    "dispatch_probe",
    "sample_step_tokens",
    "commit_kv_proj",
    "build_step_local",
    "softmax_temp",
    "select_mixed_accept",
    "cap_correct_len",
    "finalize_accept_lens",
    "accept_greedy",
    "accept_sampling",
    "schedule_verify_lens",
    "build_ragged_verify_window",
    "compute_window_gather",
    "build_swa_page_indices",
    "build_block_seq_lens",
    "compact_row_index",
    "compact_verify_ids",
    "scatter_compact_to_strided",
    "build_commit_inject_layout",
    "build_out_tokens",
    "build_qo_indptr_shared",
    "padded_to_bucket_shared",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--case", choices=CASE_NAMES)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--no-subprocess",
        action="store_true",
        help="Run all tests in one process (faster, but a device error may poison later cases).",
    )
    return parser.parse_args()


def init_device(device_name: str):
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for this test") from exc

    device = torch.device(device_name)
    if device.type != "npu":
        raise ValueError(f"This script is NPU-only, got --device={device_name!r}")
    if hasattr(torch, "npu"):
        torch.npu.set_device(device)
    torch.manual_seed(1234)
    return torch, device


def sync(torch, device) -> None:
    module = torch.get_device_module(device)
    if hasattr(module, "synchronize"):
        module.synchronize()


def first_tensor(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def flatten_result(value: Any, prefix: str = "result") -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if dataclasses.is_dataclass(value):
        out = {}
        for field in dataclasses.fields(value):
            out.update(flatten_result(getattr(value, field.name), f"{prefix}.{field.name}"))
        return out
    struct_fields = getattr(type(value), "__struct_fields__", None)
    if struct_fields is not None:
        out = {}
        for name in struct_fields:
            out.update(flatten_result(getattr(value, name), f"{prefix}.{name}"))
        return out
    if isinstance(value, (tuple, list)):
        out = {}
        for index, item in enumerate(value):
            out.update(flatten_result(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def compare_results(torch, expected: Any, actual: Any) -> tuple[bool, str]:
    expected_flat = flatten_result(expected)
    actual_flat = flatten_result(actual)
    if expected_flat.keys() != actual_flat.keys():
        return False, (
            f"result fields differ: expected={sorted(expected_flat)} "
            f"actual={sorted(actual_flat)}"
        )
    for name in expected_flat:
        lhs, rhs = expected_flat[name], actual_flat[name]
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
                return False, (
                    f"{name}: metadata differs: "
                    f"{tuple(lhs.shape)}/{lhs.dtype} vs {tuple(rhs.shape)}/{rhs.dtype}"
                )
            lhs_cpu = lhs.detach().cpu()
            rhs_cpu = rhs.detach().cpu()
            if lhs.dtype.is_floating_point:
                if not torch.allclose(lhs_cpu, rhs_cpu, rtol=2e-3, atol=2e-3):
                    diff = (lhs_cpu.float() - rhs_cpu.float()).abs().max().item()
                    return False, f"{name}: max_abs_diff={diff}"
            elif not torch.equal(lhs_cpu, rhs_cpu):
                return False, f"{name}: tensor values differ"
        elif lhs != rhs:
            return False, f"{name}: {lhs!r} != {rhs!r}"
    return True, ""


def clone_kwargs(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_kwargs(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_kwargs(item) for item in value)
    if isinstance(value, list):
        return [clone_kwargs(item) for item in value]
    return value


def standard_case(
    *,
    torch,
    device,
    op,
    kwargs: dict[str, Any],
    expected_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    from sglang.srt.speculative.dspark_components.kernels.dispatch import (
        inputs_on_cuda,
    )

    tensor = first_tensor(kwargs)
    direct_is_cuda_dispatch = op.__name__ in {
        "SampleStepTokens",
        "CommitKvProj",
        "BuildQoIndptr",
        "PaddedToBucket",
    }
    if tensor is None:
        route_to_triton = None
    elif direct_is_cuda_dispatch:
        route_to_triton = bool(tensor.is_cuda)
    else:
        route_to_triton = bool(inputs_on_cuda(**kwargs))
    base = {
        "device": str(tensor.device) if tensor is not None else None,
        "device_type": tensor.device.type if tensor is not None else None,
        "tensor_is_cuda": bool(tensor.is_cuda) if tensor is not None else None,
        "dispatch_reports_cuda": route_to_triton,
        "dispatch_rule": ".is_cuda" if direct_is_cuda_dispatch else "inputs_on_cuda",
    }

    ref = expected_fn or op.torch
    try:
        torch.manual_seed(1234)
        expected = ref(**clone_kwargs(kwargs))
        sync(torch, device)
    except Exception as exc:
        return {
            **base,
            "status": "REFERENCE_GAP",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    try:
        torch.manual_seed(1234)
        actual = op.execute(**clone_kwargs(kwargs))
        sync(torch, device)
    except Exception as exc:
        return {
            **base,
            "status": "NEEDS_MODIFICATION",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    equal, detail = compare_results(torch, expected, actual)
    if not equal:
        return {**base, "status": "MISMATCH", "error": detail}
    if tensor is not None and tensor.device.type == "npu" and route_to_triton:
        return {
            **base,
            "status": "MISROUTED_TO_TRITON",
            "error": "NPU tensor was classified as CUDA even though outputs matched.",
        }
    return {**base, "status": "PASS", "error": ""}


def make_layout(torch, device):
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

    verify_lens = torch.tensor([3, 2], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    return RaggedVerifyLayout(
        verify_lens=verify_lens,
        graph_num_tokens=8,
        extend_start_loc=qo_indptr[:-1].clone(),
        qo_indptr_device=qo_indptr,
        verify_lens_cpu=[3, 2],
        total_verify_tokens=5,
    )


def build_case(name: str, torch, device):
    from sglang.srt.speculative.dspark_components.kernels.dspark_accept import (
        AcceptGreedy,
        AcceptSampling,
        CapCorrectLen,
        FinalizeAcceptLens,
        SelectMixedAccept,
        SoftmaxTemp,
    )
    from sglang.srt.speculative.dspark_components.kernels.dspark_attn_metadata import (
        BuildBlockSeqLensCausal,
        BuildDsparkSwaPageIndices,
        ComputeDsparkWindowGather,
    )
    from sglang.srt.speculative.dspark_components.kernels.dspark_draft_model import (
        BuildStepLocal,
        CommitKvProj,
        SampleStepTokens,
    )
    from sglang.srt.speculative.dspark_components.kernels.dspark_schedule import (
        ScheduleVerifyLensTopk,
    )
    from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
        BuildCommitInjectLayout,
        BuildOutTokens,
        BuildRaggedVerifyWindow,
        CompactRowIndex,
        CompactVerifyIds,
        ScatterCompactToStrided,
    )

    if name == "sample_step_tokens":
        return SampleStepTokens, {
            "step_logits": torch.randn(2, 32, dtype=torch.bfloat16, device=device),
            "temperatures": torch.tensor([1.0, 0.7], device=device),
            "greedy_mask": torch.tensor([True, False], device=device),
            "exp_noise": torch.rand(2, 32, device=device).clamp_min_(1e-4),
        }
    if name == "commit_kv_proj":
        class TupleLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(8, 16, dtype=torch.bfloat16, device=device)
                )

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight), None

        return CommitKvProj, {
            "main_x": torch.randn(4, 16, dtype=torch.bfloat16, device=device),
            "wkv_linears": [TupleLinear(), TupleLinear()],
        }
    if name == "build_step_local":
        return BuildStepLocal, {
            "bias": torch.randn(2, 7, dtype=torch.float32, device=device),
            "base_local": torch.randn(2, 12, dtype=torch.float32, device=device),
        }
    if name == "softmax_temp":
        return SoftmaxTemp, {
            "logits": torch.randn(6, 64, dtype=torch.bfloat16, device=device),
            "temperatures": torch.tensor([0.8, 1.2], device=device),
            "rows_per_request": 3,
        }
    if name == "select_mixed_accept":
        return SelectMixedAccept, {
            "greedy_mask": torch.tensor([True, False], device=device),
            "greedy_len": torch.tensor([2, 1], dtype=torch.int64, device=device),
            "greedy_bonus": torch.tensor([11, 12], dtype=torch.int64, device=device),
            "greedy_trim": torch.tensor([0, 1], dtype=torch.int64, device=device),
            "sampling_len": torch.tensor([1, 2], dtype=torch.int64, device=device),
            "sampling_bonus": torch.tensor([21, 22], dtype=torch.int64, device=device),
            "sampling_trim": torch.tensor([1, 0], dtype=torch.int64, device=device),
        }
    if name == "cap_correct_len":
        return CapCorrectLen, {
            "correct_len": torch.tensor([4, 2], dtype=torch.int64, device=device),
            "verify_lens": torch.tensor([3, 4], dtype=torch.int32, device=device),
        }
    if name == "finalize_accept_lens":
        return FinalizeAcceptLens, {
            "correct_len": torch.tensor([2, 1], dtype=torch.int64, device=device),
            "cap_trim_lens": torch.tensor([1, 0], dtype=torch.int64, device=device),
            "prefix_lens": torch.tensor([10, 20], dtype=torch.int64, device=device),
        }
    if name == "accept_greedy":
        bs, width, vocab = 2, 4, 32
        candidates = torch.tensor(
            [[3, 4, 5, 6], [7, 8, 9, 10]], dtype=torch.int64, device=device
        )
        predicts = candidates.clone()
        predicts[:, -1] += 1
        logits = torch.full((bs, width, vocab), -5.0, device=device)
        logits.scatter_(2, predicts.unsqueeze(-1), 5.0)
        return AcceptGreedy, {
            "candidates": candidates,
            "target_logits": logits.view(bs * width, vocab),
            "verify_num_draft_tokens": width,
            "cutoff_verify_lens": torch.tensor([3, 4], dtype=torch.int32, device=device),
        }
    if name == "accept_sampling":
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

        bs, gamma, width, vocab = 2, 2, 3, 16
        candidates = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        target_logits = torch.randn(bs * width, vocab, device=device)
        draft_probs = torch.softmax(torch.randn(bs, gamma, vocab, device=device), -1)
        draft_input = DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device),
            topk_index=torch.empty((bs, 0), dtype=torch.int64, device=device),
            bonus_tokens=torch.zeros(bs, dtype=torch.int64, device=device),
            new_seq_lens=torch.ones(bs, dtype=torch.int64, device=device),
            hidden_states=torch.empty((bs, 0), device=device),
        )
        sampling_info = SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            temperatures=torch.ones(bs, device=device),
        )
        return AcceptSampling, {
            "candidates": candidates,
            "target_logits": target_logits,
            "draft_probs": draft_probs,
            "sampling_info": sampling_info,
            "draft_input": draft_input,
            "gamma": gamma,
            "verify_num_draft_tokens": width,
            "cutoff_verify_lens": None,
        }
    if name == "schedule_verify_lens":
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            DSparkScheduleConfig,
        )

        return ScheduleVerifyLensTopk, {
            "confidence": torch.tensor(
                [[0.9, 0.8, 0.5], [0.7, 0.6, 0.4]], device=device
            ),
            "budget": 3,
            "cfg": DSparkScheduleConfig(gamma=3, min_verify_len=1),
        }
    if name == "build_ragged_verify_window":
        req_to_token = torch.arange(64, dtype=torch.int64, device=device).view(2, 32)
        batch = SimpleNamespace(
            seq_lens=torch.tensor([4, 6], dtype=torch.int64, device=device),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=device),
        )
        model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
        )
        return BuildRaggedVerifyWindow, {
            "batch": batch,
            "layout": make_layout(torch, device),
            "draft_block_ids": torch.tensor(
                [[100, 0, 0], [200, 0, 0]], dtype=torch.int64, device=device
            ),
            "draft_tokens": torch.tensor(
                [[101, 102, 103], [201, 202, 203]],
                dtype=torch.int64,
                device=device,
            ),
            "bs": 2,
            "device": str(device),
            "verify_num_draft_tokens": 4,
            "model_runner": model_runner,
        }
    if name == "compute_window_gather":
        return ComputeDsparkWindowGather, {
            "seq_lens_casual": torch.tensor(
                [11, 12, 13, 14, 7, 8, 9, 10], dtype=torch.int32, device=device
            ),
            "req_pool_indices_repeated": torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64, device=device
            ),
            "block_size": 4,
            "swa_window": 8,
        }
    if name == "build_swa_page_indices":
        req_to_token = torch.arange(64, dtype=torch.int64, device=device).view(2, 32)
        mapping = torch.arange(128, dtype=torch.int32, device=device)
        return BuildDsparkSwaPageIndices, {
            "req_to_token": req_to_token,
            "full_to_swa_mapping": mapping,
            "req_pool_indices_per_request": torch.tensor([0, 1], device=device),
            "offsets": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
                dtype=torch.int64,
                device=device,
            ),
            "invalid": torch.zeros((2, 8), dtype=torch.bool, device=device),
            "out_loc": torch.tensor([20, 21, 22, 23, 52, 53, 54, 55], device=device),
            "context_lens": torch.tensor([8, 6], dtype=torch.int32, device=device),
            "block_size": 4,
            "swa_window": 8,
            "page_index_aligned_size": 8,
        }
    if name == "build_block_seq_lens":
        return BuildBlockSeqLensCausal, {
            "seq_lens": torch.tensor([10, 20], dtype=torch.int64, device=device),
            "block_size": 4,
            "device": device,
        }
    if name == "compact_row_index":
        return CompactRowIndex, {
            "verify_lens": torch.tensor([3, 2], dtype=torch.int32, device=device),
            "padded_total": 8,
            "device": device,
        }
    if name == "compact_verify_ids":
        return CompactVerifyIds, {
            "draft_block_ids": torch.tensor(
                [[100, 0, 0], [200, 0, 0]], dtype=torch.int64, device=device
            ),
            "draft_tokens": torch.tensor(
                [[101, 102, 103], [201, 202, 203]], dtype=torch.int64, device=device
            ),
            "layout": make_layout(torch, device),
            "device": str(device),
        }
    if name == "scatter_compact_to_strided":
        return ScatterCompactToStrided, {
            "compact": torch.arange(16, dtype=torch.float32, device=device).view(8, 2),
            "layout": make_layout(torch, device),
            "fill_value": -1.0,
            "verify_num_draft_tokens": 4,
        }
    if name == "build_commit_inject_layout":
        req_to_token = torch.arange(64, dtype=torch.int64, device=device).view(2, 32)
        mapping = torch.arange(128, dtype=torch.int32, device=device)
        return BuildCommitInjectLayout, {
            "req_pool_indices": torch.tensor([0, 1], dtype=torch.int64, device=device),
            "req_to_token": req_to_token,
            "prefix_lens": torch.tensor([4, 6], dtype=torch.int64, device=device),
            "block_pos_offsets": torch.arange(4, dtype=torch.int64, device=device),
            "full_to_swa_mapping": mapping,
            "commit_lens": torch.tensor([3, 2], dtype=torch.int32, device=device),
            "stride": 4,
        }
    if name == "build_out_tokens":
        return BuildOutTokens, {
            "draft_tokens": torch.tensor(
                [[11, 12, 13], [21, 22, 23]], dtype=torch.int64, device=device
            ),
            "correct_len": torch.tensor([2, 1], dtype=torch.int64, device=device),
            "bonus": torch.tensor([99, 88], dtype=torch.int64, device=device),
            "verify_num_draft_tokens": 4,
            "gamma": 3,
        }
    if name == "build_qo_indptr_shared":
        from sglang.srt.speculative.ragged_verify_kernels import BuildQoIndptr

        return BuildQoIndptr, {
            "verify_lens": torch.tensor([3, 2], dtype=torch.int32, device=device)
        }
    if name == "padded_to_bucket_shared":
        from sglang.srt.speculative.ragged_verify_kernels import PaddedToBucket

        return PaddedToBucket, {
            "verify_lens": torch.tensor([3, 2], dtype=torch.int32, device=device),
            "graph_num_tokens": 8,
            "bs": 2,
            "padded_bs": 3,
        }
    raise KeyError(name)


def run_one(name: str, device_name: str) -> dict[str, Any]:
    torch, device = init_device(device_name)

    if name == "dispatch_probe":
        from sglang.srt.speculative.dspark_components.kernels.dispatch import (
            inputs_on_cuda,
        )

        tensor = torch.ones(1, device=device)
        reports_cuda = bool(inputs_on_cuda(tensor))
        return {
            "status": "NEEDS_MODIFICATION" if reports_cuda else "PASS",
            "device": str(tensor.device),
            "device_type": tensor.device.type,
            "tensor_is_cuda": bool(tensor.is_cuda),
            "dispatch_reports_cuda": reports_cuda,
            "error": (
                "inputs_on_cuda() classifies an NPU tensor as CUDA."
                if reports_cuda
                else ""
            ),
        }

    op, kwargs = build_case(name, torch, device)
    return standard_case(torch=torch, device=device, op=op, kwargs=kwargs)


def emit_child_result(name: str, device_name: str) -> int:
    try:
        result = run_one(name, device_name)
    except Exception as exc:
        result = {
            "status": "HARNESS_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    result["case"] = name
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
    return 0


def run_parent(args: argparse.Namespace) -> int:
    results = []
    for name in CASE_NAMES:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--device",
            args.device,
            "--case",
            name,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PYTHON_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
            marker = next(
                (line for line in reversed(proc.stdout.splitlines()) if line.startswith("RESULT_JSON:")),
                None,
            )
            if marker is None:
                result = {
                    "case": name,
                    "status": "PROCESS_FAILED",
                    "error": f"exit={proc.returncode}; output={proc.stdout[-2000:]}",
                }
            else:
                result = json.loads(marker[len("RESULT_JSON:") :])
        except subprocess.TimeoutExpired as exc:
            result = {
                "case": name,
                "status": "TIMEOUT",
                "error": f"exceeded {args.timeout}s; output={(exc.stdout or '')[-1000:]}",
            }
        results.append(result)
        route = result.get("dispatch_reports_cuda")
        route_text = " triton-route" if route else ""
        print(
            f"{name:32s} {result['status']:24s}{route_text} "
            f"{result.get('error', '')}"
        )

    print("\nSummary")
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    for status, count in sorted(counts.items()):
        print(f"  {status:24s}: {count}")

    print("\nMachine-readable results")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    bad = sum(result["status"] != "PASS" for result in results)
    return 1 if bad else 0


def run_in_process(args: argparse.Namespace) -> int:
    results = []
    for name in CASE_NAMES:
        result = run_one(name, args.device)
        result["case"] = name
        results.append(result)
        print(f"{name:32s} {result['status']:24s} {result.get('error', '')}")
    return 1 if any(item["status"] != "PASS" for item in results) else 0


def main() -> int:
    args = parse_args()
    if args.case:
        return emit_child_result(args.case, args.device)
    if args.no_subprocess:
        return run_in_process(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
