from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import support_triton
from sglang.srt.utils.common import ceil_align
from sglang.srt.distributed.parallel_state import (
    get_context_parallel_world_size,
    get_context_parallel_rank,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
    reqs: list["Req"] | None = None,
):
    # Check if CP is enabled
    enable_cp = get_context_parallel_world_size() > 1
    cp_rank = get_context_parallel_rank() if enable_cp else -1

    if support_triton(get_global_server_args().attention_backend) and not enable_cp:
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            device=req_to_token_pool.device,
            dtype=torch.uint64,
        )
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )

            if enable_cp:
                if reqs is None:
                    raise ValueError("reqs must be provided when CP mode is enabled")
                cp_metadata = reqs[i].cp_metadata
                actual_seq_len = cp_metadata.actual_seq_len
                out_offset = pt
                written_blocks = []

                for block_idx in cp_metadata.zigzag_index:
                    block_size = cp_metadata.split_list[block_idx]

                    # Calculate token range of block_idx
                    block_token_start = 0
                    for j in range(block_idx):
                        block_token_start += cp_metadata.split_list[j]
                    block_token_end = block_token_start + block_size

                    # Skip if block exceeds actual sequence length
                    if block_token_start >= actual_seq_len:
                        break

                    # Calculate write range
                    extend_block_start = block_token_start
                    extend_block_end = min(block_token_end, actual_seq_len)

                    # Calculate number of KV indices to write
                    write_size = extend_block_end - extend_block_start

                    # Write to req_to_token_pool
                    req_to_token_pool.write(
                        (req_idx, slice(extend_block_start, extend_block_end)),
                        out_cache_loc[out_offset : out_offset + write_size],
                    )

                    written_blocks.append({
                        'block_idx': block_idx,
                        'token_range': (extend_block_start, extend_block_end),
                        'write_size': write_size,
                        'kv_indices_sample': out_cache_loc[out_offset : out_offset + min(5, write_size)].cpu().tolist()
                    })
                    out_offset += write_size
                
                # 验证点7: 写入的KV indices数量
                written_indices = req_to_token_pool.req_to_token[req_idx, :actual_seq_len].cpu().tolist()
                non_zero_count = len([x for x in written_indices if x != 0 and x != -1])
                expected_written = sum(block['write_size'] for block in written_blocks)
                is_written_count_correct = (non_zero_count == expected_written)
                print(f"[CP_MEM_VERIFY] cp_rank={cp_rank} | 验证点7: 写入KV indices数量 | "
                      f"req_id={reqs[i].rid if reqs else 'N/A'} | "
                      f"written_count={non_zero_count} | expected={expected_written} | "
                      f"是否符合预期={'✓' if is_written_count_correct else '✗'}")
                
                # 验证点8: 写入的token范围正确性
                all_ranges_valid = True
                for block_info in written_blocks:
                    start, end = block_info['token_range']
                    if not (0 <= start < actual_seq_len) or not (0 < end <= actual_seq_len):
                        all_ranges_valid = False
                        break
                print(f"[CP_MEM_VERIFY] cp_rank={cp_rank} | 验证点8: 写入token范围 | "
                      f"req_id={reqs[i].rid if reqs else 'N/A'} | actual_seq_len={actual_seq_len} | "
                      f"所有块范围是否在[0, {actual_seq_len})内={'✓' if all_ranges_valid else '✗'}")
                
                # 验证点9: 块覆盖范围完整性
                total_written_range = sum(
                    end - start for block_info in written_blocks
                    for start, end in [block_info['token_range']]
                )
                expected_range = sum(
                    cp_metadata.split_list[j] for j in cp_metadata.zigzag_index
                )
                is_range_correct = (total_written_range <= expected_range)
                print(f"[CP_MEM_VERIFY] cp_rank={cp_rank} | 验证点9: 块覆盖范围 | "
                      f"req_id={reqs[i].rid if reqs else 'N/A'} | "
                      f"total_written_range={total_written_range} | expected_range={expected_range} | "
                      f"是否符合预期={'✓' if is_range_correct else '✗'} (written <= expected)")
                
                # 验证点10: req_to_token_pool写入位置正确性
                # 验证每个块写入的位置是否与预期的块位置匹配
                out_offset_check = pt
                all_positions_correct = True
                all_kv_match = True
                position_details = []
                
                for block_info in written_blocks:
                    block_idx = block_info['block_idx']
                    start, end = block_info['token_range']
                    write_size = block_info['write_size']
                    
                    # 计算这个块在原始序列中的预期位置
                    expected_block_start = 0
                    for j in range(block_idx):
                        expected_block_start += cp_metadata.split_list[j]
                    expected_block_end = expected_block_start + cp_metadata.split_list[block_idx]
                    
                    # 实际写入的位置应该等于预期的块位置（受actual_seq_len限制）
                    actual_write_start = start
                    actual_write_end = end
                    expected_write_start = max(0, expected_block_start)
                    expected_write_end = min(expected_block_end, actual_seq_len)
                    
                    is_position_correct = (actual_write_start == expected_write_start and 
                                          actual_write_end == expected_write_end)
                    
                    if not is_position_correct:
                        all_positions_correct = False
                    
                    # 验证写入的KV indices是否与分配的out_cache_loc一致
                    written_kv_indices = req_to_token_pool.req_to_token[req_idx, actual_write_start:actual_write_end].cpu()
                    expected_kv_indices = out_cache_loc[out_offset_check:out_offset_check + write_size].cpu()
                    
                    # 检查是否匹配（排除-1和0的情况）
                    written_valid = written_kv_indices[written_kv_indices > 0]
                    expected_valid = expected_kv_indices[expected_kv_indices > 0]
                    is_kv_match = len(written_valid) == len(expected_valid)
                    if is_kv_match and len(written_valid) > 0:
                        is_kv_match = torch.allclose(written_valid.float(), expected_valid.float(), atol=1e-5)
                    
                    if not is_kv_match:
                        all_kv_match = False
                    
                    position_details.append({
                        'block_idx': block_idx,
                        'expected_range': (expected_block_start, expected_block_end),
                        'actual_write_range': (actual_write_start, actual_write_end),
                        'expected_write_range': (expected_write_start, expected_write_end),
                        'is_position_correct': is_position_correct,
                        'is_kv_match': is_kv_match,
                    })
                    
                    out_offset_check += write_size
                
                print(f"[CP_MEM_VERIFY] cp_rank={cp_rank} | 验证点10: req_to_token_pool写入位置 | "
                      f"req_id={reqs[i].rid if reqs else 'N/A'} | "
                      f"所有块写入位置是否正确={'✓' if all_positions_correct else '✗'} | "
                      f"所有KV indices是否匹配={'✓' if all_kv_match else '✗'} | "
                      f"position_details={position_details}")
            else:
                req_to_token_pool.write(
                    (req_idx, slice(prefix_len, seq_len)),
                    out_cache_loc[pt : pt + extend_len],
                )

            pt += extend_len


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    if (
        get_global_server_args().attention_backend != "ascend"
        and get_global_server_args().attention_backend != "torch_native"
    ):
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
        factor = (
            MAMBA_STATE_PER_REQ_PREFIX_CACHE
            if tree_cache.supports_mamba()
            else MAMBA_STATE_PER_REQ_NO_CACHE
        )
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        req_pool_indices_device: request pool indices at a device tensor
        req_pool_indices: request pool indices as list
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if batch.tree_cache.page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        
        seq_lens_for_alloc = batch.seq_lens
        seq_lens_cpu_for_alloc = batch.seq_lens_cpu
        
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens_for_alloc,
            seq_lens_cpu=seq_lens_cpu_for_alloc,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
        )
    
    # 验证点5: 内存分配大小正确性
    enable_cp = get_context_parallel_world_size() > 1
    if enable_cp:
        cp_rank = get_context_parallel_rank()
        allocated_size = len(out_cache_loc)
        expected_size = batch.extend_num_tokens
        is_alloc_correct = (allocated_size == expected_size)
        print(f"[CP_MEM_VERIFY] cp_rank={cp_rank} | 验证点5: 内存分配大小 | "
              f"allocated={allocated_size} | expected={expected_size} | "
              f"是否符合预期={'✓' if is_alloc_correct else '✗'}")
        
        # 验证点6: 每个请求分配的内存大小
        if batch.reqs is not None and len(batch.reqs) > 0:
            extend_lens_list = extend_lens_cpu.tolist()
            total_allocated = 0
            for i, req in enumerate(batch.reqs):
                if i < len(extend_lens_list):
                    req_extend = extend_lens_list[i]
                    start_idx = sum(extend_lens_list[:i])
                    end_idx = start_idx + req_extend
                    allocated_for_req = end_idx - start_idx
                    is_req_alloc_correct = (allocated_for_req == req_extend)
                    req_id = req.rid if hasattr(req, 'rid') else f"req_{i}"
                    print(f"[CP_MEM_VERIFY] cp_rank={cp_rank} | 验证点6: 请求内存分配 | "
                          f"req_id={req_id} | "
                          f"allocated={allocated_for_req} | expected={req_extend} | "
                          f"是否符合预期={'✓' if is_req_alloc_correct else '✗'}")
                    total_allocated += allocated_for_req
            
            is_total_alloc_correct = (total_allocated == expected_size)
            print(f"[CP_MEM_VERIFY] cp_rank={cp_rank} | 验证点6: 总分配内存 | "
                  f"total_allocated={total_allocated} | expected={expected_size} | "
                  f"是否符合预期={'✓' if is_total_alloc_correct else '✗'}")

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
        batch.reqs,
    )

    return out_cache_loc, req_pool_indices_device, req_pool_indices


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    bs = batch.seq_lens.shape[0]

    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        seq_lens_next = batch.seq_lens + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    tree_cache.cache_finished_req(req, is_insert=is_insert)

    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    if spec_algo is None:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache) -> str:
    token_to_kv_pool_allocator = tree_cache.token_to_kv_pool_allocator
    if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
        full_available_size = token_to_kv_pool_allocator.full_available_size()
        swa_available_size = token_to_kv_pool_allocator.swa_available_size()
        full_evictable_size = tree_cache.full_evictable_size()
        swa_evictable_size = tree_cache.swa_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
            f"Full LRU list evictable size: {tree_cache.full_lru_list_evictable_size()}\n"
            f"SWA LRU list evictable size: {tree_cache.swa_lru_list_evictable_size()}\n"
        )
    else:
        available_size = token_to_kv_pool_allocator.available_size()
        evictable_size = tree_cache.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
