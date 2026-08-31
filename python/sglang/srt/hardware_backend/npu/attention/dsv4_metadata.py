"""Triton kernels for DeepSeek-V4 Ascend metadata refresh."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

from sglang.srt.hardware_backend.npu.utils import is_npu_arch35


@triton.jit
def _step_compress_kernel(
    seq_lens_ptr,
    full_loc_ptr,
    swa_loc_ptr,
    c4_loc_ptr,
    c128_loc_ptr,
    full_out_ptr,
    swa_out_ptr,
    c4_out_ptr,
    c128_out_ptr,
    raw_bs,
    TOPK: tl.constexpr,
    NUM_STEPS: tl.constexpr,
    STEP_ID: tl.constexpr,
    HAS_C4: tl.constexpr,
    HAS_C128: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Stable compact for one draft step without materializing masks/offsets."""
    lane = tl.arange(0, BLOCK_SIZE)
    lane_mask = lane < raw_bs * TOPK
    batch = lane // TOPK
    seq_len = tl.load(seq_lens_ptr + batch, mask=lane_mask, other=0).to(tl.int32)
    source_lane = lane * NUM_STEPS + STEP_ID
    full_loc = tl.load(full_loc_ptr + source_lane, mask=lane_mask)
    swa_loc = tl.load(swa_loc_ptr + source_lane, mask=lane_mask)
    tl.store(full_out_ptr + lane, full_loc, mask=lane_mask)
    tl.store(swa_out_ptr + lane, swa_loc, mask=lane_mask)

    if HAS_C4:
        # ratio=4: shifts/masks keep the address path vectorized on Ascend.
        total = ((seq_len + NUM_STEPS) >> 2) - (seq_len >> 2)
        before = ((seq_len + STEP_ID) >> 2) - (seq_len >> 2)
        selected = (((seq_len + STEP_ID + 1) & 3) == 0) & lane_mask
        src_offset = tl.cumsum(total, axis=0) - total + before
        dst_offset = tl.cumsum(selected.to(tl.int32), axis=0) - selected.to(tl.int32)
        value = tl.load(c4_loc_ptr + src_offset, mask=selected)
        tl.store(c4_out_ptr + dst_offset, value, mask=selected)

    if HAS_C128:
        total = ((seq_len + NUM_STEPS) >> 7) - (seq_len >> 7)
        before = ((seq_len + STEP_ID) >> 7) - (seq_len >> 7)
        selected = (((seq_len + STEP_ID + 1) & 127) == 0) & lane_mask
        src_offset = tl.cumsum(total, axis=0) - total + before
        dst_offset = tl.cumsum(selected.to(tl.int32), axis=0) - selected.to(tl.int32)
        value = tl.load(c128_loc_ptr + src_offset, mask=selected)
        tl.store(c128_out_ptr + dst_offset, value, mask=selected)


def _step_sizes_from_cpu(
    seq_lens_cpu,
    *,
    raw_bs: int,
    topk: int,
    num_steps: int,
    step_id: int,
) -> tuple[int, int, int, int]:
    if isinstance(seq_lens_cpu, torch.Tensor):
        if seq_lens_cpu.device.type != "cpu":
            raise ValueError("step_compress requires a CPU seq_lens mirror")
        seq_lens = seq_lens_cpu[:raw_bs].tolist()
    else:
        seq_lens = list(seq_lens_cpu[:raw_bs])

    c4_size = topk * sum((int(seq_len) + step_id + 1) % 4 == 0 for seq_len in seq_lens)
    c128_size = topk * sum(
        (int(seq_len) + step_id + 1) % 128 == 0 for seq_len in seq_lens
    )
    c4_source_size = topk * sum(
        (int(seq_len) + num_steps) // 4 - int(seq_len) // 4 for seq_len in seq_lens
    )
    c128_source_size = topk * sum(
        (int(seq_len) + num_steps) // 128 - int(seq_len) // 128 for seq_len in seq_lens
    )
    return c4_size, c128_size, c4_source_size, c128_source_size


def step_compress(
    full_loc: torch.Tensor,
    swa_loc: torch.Tensor,
    c4_loc: Optional[torch.Tensor],
    c128_loc: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    seq_lens_cpu,
    *,
    raw_bs: int,
    topk: int,
    num_steps: int,
    step_id: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Extract full/SWA/C4/C128 locations for one step in one Triton launch.

    ``seq_lens_cpu`` only determines the exact output shapes. It is the existing
    host mirror used by draft graph replay, so this does not introduce a device
    synchronization. The location values and stable compaction stay on device.
    """
    if not 0 <= step_id < num_steps:
        raise ValueError(f"step_id must be in [0, {num_steps}), got {step_id}")
    if raw_bs < 0 or topk <= 0 or num_steps <= 0:
        raise ValueError(
            "step_compress requires raw_bs >= 0, topk > 0, and num_steps > 0"
        )
    step_width = raw_bs * topk
    required_full_size = step_width * num_steps
    if full_loc.numel() < required_full_size or swa_loc.numel() < required_full_size:
        raise RuntimeError(
            "full/SWA step location buffer is too short: "
            f"full={full_loc.numel()}, swa={swa_loc.numel()}, "
            f"required={required_full_size}"
        )

    has_c4_source = c4_loc is not None and c4_loc.numel() > 0
    has_c128_source = c128_loc is not None and c128_loc.numel() > 0
    if not has_c4_source and not has_c128_source:
        c4_size = c128_size = c4_source_size = c128_source_size = 0
    else:
        c4_size, c128_size, c4_source_size, c128_source_size = _step_sizes_from_cpu(
            seq_lens_cpu,
            raw_bs=raw_bs,
            topk=topk,
            num_steps=num_steps,
            step_id=step_id,
        )
        if not has_c4_source:
            c4_size = c4_source_size = 0
        if not has_c128_source:
            c128_size = c128_source_size = 0
    for ratio, loc, required in (
        (4, c4_loc, c4_source_size),
        (128, c128_loc, c128_source_size),
    ):
        if loc is not None and loc.numel() > 0 and loc.numel() < required:
            raise RuntimeError(
                f"C{ratio} step location buffer is too short: "
                f"{loc.numel()} < {required}"
            )

    full_out = full_loc.new_empty((step_width,))
    swa_out = swa_loc.new_empty((step_width,))
    c4_out = None if c4_loc is None else c4_loc.new_empty((c4_size,))
    c128_out = None if c128_loc is None else c128_loc.new_empty((c128_size,))
    has_c4 = c4_out is not None and c4_size > 0
    has_c128 = c128_out is not None and c128_size > 0
    if step_width == 0:
        return full_out, swa_out, c4_out, c128_out

    lanes = raw_bs * topk
    block_size = triton.next_power_of_2(max(1, lanes))
    # Compile-time-disabled branches do not dereference their placeholder pointers.
    c4_loc_arg = c4_loc if c4_loc is not None else seq_lens
    c128_loc_arg = c128_loc if c128_loc is not None else seq_lens
    c4_out_arg = c4_out if c4_out is not None else seq_lens
    c128_out_arg = c128_out if c128_out is not None else seq_lens
    launch_options = {"multibuffer": True}
    if is_npu_arch35():
        launch_options.update(enable_vf_fusion=True, enable_flatten=True)
    _step_compress_kernel[(1,)](
        seq_lens,
        full_loc,
        swa_loc,
        c4_loc_arg,
        c128_loc_arg,
        full_out,
        swa_out,
        c4_out_arg,
        c128_out_arg,
        raw_bs,
        TOPK=topk,
        NUM_STEPS=num_steps,
        STEP_ID=step_id,
        HAS_C4=has_c4,
        HAS_C128=has_c128,
        BLOCK_SIZE=block_size,
        **launch_options,
    )
    return full_out, swa_out, c4_out, c128_out


@triton.jit
def _refresh_graph_explicit_state_block_kernel(
    dst_ptr,
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_indices_ptr,
    start_pos_ptr,
    seqused_ptr,
    cu_seqlens_ptr,
    batch_size,
    req_to_token_stride,
    mapping_size,
    NUM_PROGRAMS: tl.constexpr,
    WIDTH: tl.constexpr,
    HISTORY_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    RING_SIZE: tl.constexpr,
    SWA_PAGE_SIZE: tl.constexpr,
    DUMMY_STATE_LOC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    column_mask = columns < WIDTH

    for row in tl.range(pid, batch_size, NUM_PROGRAMS):
        req = tl.load(req_pool_indices_ptr + row).to(tl.int32)
        start_pos = tl.load(start_pos_ptr + row).to(tl.int32)
        seqused = tl.load(seqused_ptr + row).to(tl.int32)
        capacity = (
            tl.load(cu_seqlens_ptr + row + 1) - tl.load(cu_seqlens_ptr + row)
        ).to(tl.int32)
        positions = start_pos - HISTORY_SIZE + columns
        valid = (
            column_mask
            & (seqused > 0)
            & (positions >= 0)
            & (columns < HISTORY_SIZE + capacity)
        )

        if RATIO == 4:
            safe_positions = tl.maximum(
                0, tl.minimum(positions, req_to_token_stride - 1)
            )
            full_loc = tl.load(
                req_to_token_ptr + req * req_to_token_stride + safe_positions,
                mask=valid,
                other=-1,
            ).to(tl.int32)
            # PyTorch maps a -1 request-table entry to the mapping sentinel.
            mapping_index = tl.where(full_loc < 0, mapping_size + full_loc, full_loc)
            swa_loc = tl.load(full_to_swa_ptr + mapping_index, mask=valid, other=-1)
            state_loc = (swa_loc // SWA_PAGE_SIZE) * RING_SIZE + (
                swa_loc & (RING_SIZE - 1)
            )
            valid = valid & (swa_loc >= 0)
        else:
            state_loc = req * RING_SIZE + (positions % RING_SIZE)

        output = tl.where(valid, state_loc, DUMMY_STATE_LOC).to(tl.int32)
        tl.store(dst_ptr + row * WIDTH + columns, output, mask=column_mask)


def _refresh_graph_explicit_state_block(
    dst: torch.Tensor,
    *,
    compress_ratio: int,
    state_pool,
    token_to_kv_pool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    start_pos: torch.Tensor,
    seqused: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    """Refresh one pre-A5 cache_mode=2 table directly into graph storage."""
    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported DSV4 compress ratio: {compress_ratio}")
    if dst.ndim != 2 or dst.dtype != torch.int32:
        raise ValueError("explicit state block destination must be a 2-D int32 tensor")
    if dst.shape[0] == 0:
        return

    history_size = (2 if compress_ratio == 4 else 1) * compress_ratio
    width = dst.shape[1]
    if width < history_size:
        raise ValueError(
            f"C{compress_ratio} state table width {width} is below history "
            f"size {history_size}"
        )
    ring_size = int(state_pool.ring_size)
    if ring_size <= 0 or ring_size & (ring_size - 1):
        raise ValueError(
            f"state ring size must be a positive power of two, got {ring_size}"
        )

    _, num_vector_cores = get_device_properties()
    num_programs = min(int(num_vector_cores), dst.shape[0])
    mapping = token_to_kv_pool.full_to_swa_index_mapping
    if compress_ratio == 4 and mapping is None:
        raise ValueError("C4 explicit state refresh requires full-to-SWA mapping")
    mapping_arg = mapping if mapping is not None else req_to_token
    _refresh_graph_explicit_state_block_kernel[(num_programs,)](
        dst,
        req_to_token,
        mapping_arg,
        req_pool_indices,
        start_pos,
        seqused,
        cu_seqlens,
        dst.shape[0],
        req_to_token.shape[1],
        mapping_arg.numel(),
        NUM_PROGRAMS=num_programs,
        WIDTH=width,
        HISTORY_SIZE=history_size,
        RATIO=compress_ratio,
        RING_SIZE=ring_size,
        SWA_PAGE_SIZE=int(state_pool.swa_page_size),
        DUMMY_STATE_LOC=int(state_pool.dummy_state_loc),
        BLOCK_SIZE=triton.next_power_of_2(width),
        multibuffer=True,
    )
