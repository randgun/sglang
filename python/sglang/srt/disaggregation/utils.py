from __future__ import annotations

import os
import random
from collections import deque
from contextlib import nullcontext
from enum import Enum
from itertools import accumulate
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Type, Union, overload

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa.utils import ContextParallelMetadata
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base.conn import KVArgs
    from sglang.srt.disaggregation.common.conn import (
        CommonKVBootstrapServer,
        CommonKVManager,
        CommonKVReceiver,
        CommonKVSender,
    )
    from sglang.srt.managers.schedule_batch import Req

#########################
# Constants & Enums
#########################
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


#########################
# CP Transfer Metadata
#########################


def _build_cp_block_layout(actual_seq_len: int, cp_size: int, page_size: int):
    """Build the global block layout shared by all CP ranks."""
    cp_block_num = cp_size * 2
    alignment_unit = page_size * cp_block_num
    aligned_seq_len = (
        (actual_seq_len + alignment_unit - 1) // alignment_unit
    ) * alignment_unit
    seq_len_per_block = aligned_seq_len // cp_block_num
    split_list = [seq_len_per_block] * cp_block_num
    prefix_offsets = [0] + list(accumulate(split_list))
    block_actual_lens = []
    for block_idx in range(cp_block_num):
        block_start = prefix_offsets[block_idx]
        block_end = prefix_offsets[block_idx + 1]
        if block_start >= actual_seq_len:
            block_actual_lens.append(0)
        else:
            block_actual_lens.append(min(block_end, actual_seq_len) - block_start)
    return (
        cp_block_num,
        aligned_seq_len,
        seq_len_per_block,
        split_list,
        prefix_offsets,
        block_actual_lens,
    )


def _build_cp_rank_metadata(
    actual_seq_len: int,
    cp_size: int,
    cp_rank: int,
    cp_block_num: int,
    aligned_seq_len: int,
    split_list: List[int],
    prefix_offsets: List[int],
    block_actual_lens: List[int],
    is_gqa: bool,
):
    """Build the rank-local CP metadata on top of the shared block layout."""
    bs_per_cp_group = 1  # Currently only support batch=1
    zigzag_index = list(
        range(cp_rank, cp_rank + bs_per_cp_group * cp_block_num, cp_block_num)
    ) + list(
        range(cp_block_num - cp_rank - 1, bs_per_cp_group * cp_block_num, cp_block_num)
    )

    cp_reverse_index = []
    for batch_id in range(bs_per_cp_group):
        cp_reverse_index.extend(
            list(range(batch_id, cp_block_num * bs_per_cp_group, 2 * bs_per_cp_group))
            + list(
                range(
                    (cp_block_num - 1) * bs_per_cp_group + batch_id,
                    0,
                    -2 * bs_per_cp_group,
                )
            )
        )

    rank_valid_ranges: List[Tuple[int, int]] = []
    for block_idx in zigzag_index:
        block_start = prefix_offsets[block_idx]
        block_end = prefix_offsets[block_idx + 1]
        if block_start < actual_seq_len:
            rank_valid_ranges.append((block_start, min(block_end, actual_seq_len)))

    per_rank_head_actual_token = [
        block_actual_lens[rank_idx] for rank_idx in range(cp_size)
    ]
    per_rank_tail_actual_token = [
        block_actual_lens[cp_block_num - 1 - rank_idx] for rank_idx in range(cp_size)
    ]
    head_padded_len = (
        max(per_rank_head_actual_token) if per_rank_head_actual_token else 0
    )
    tail_padded_len = (
        max(per_rank_tail_actual_token) if per_rank_tail_actual_token else 0
    )

    per_rank_actual_token = []
    reverse_split_len = []
    for rank_idx in range(cp_size):
        per_rank_actual_token.append(
            per_rank_head_actual_token[rank_idx] + per_rank_tail_actual_token[rank_idx]
        )
        reverse_split_len.extend(
            [per_rank_head_actual_token[rank_idx], per_rank_tail_actual_token[rank_idx]]
        )
    max_rank_len = head_padded_len + tail_padded_len

    head_chunk_id = cp_rank
    tail_chunk_id = cp_block_num - 1 - cp_rank
    head_start_global = prefix_offsets[head_chunk_id]
    head_end_global = prefix_offsets[head_chunk_id + 1]
    tail_start_global = prefix_offsets[tail_chunk_id]
    tail_end_global = prefix_offsets[tail_chunk_id + 1]
    head_actual_len = block_actual_lens[head_chunk_id]
    tail_actual_len = block_actual_lens[tail_chunk_id]

    return ContextParallelMetadata(
        split_list=split_list,
        max_rank_len=[max_rank_len] * cp_size,
        zigzag_index=zigzag_index,
        per_rank_actual_token=per_rank_actual_token,
        per_rank_head_actual_token=per_rank_head_actual_token,
        per_rank_tail_actual_token=per_rank_tail_actual_token,
        head_padded_len=head_padded_len,
        tail_padded_len=tail_padded_len,
        reverse_split_len=reverse_split_len,
        cp_reverse_index=cp_reverse_index,
        rank_valid_ranges=rank_valid_ranges,
        kv_len_prev=head_start_global,
        kv_len_next=tail_start_global,
        actual_seq_q_prev=head_actual_len,
        actual_seq_q_next=tail_actual_len,
        total_seq_lens=actual_seq_len,
        cp_size=cp_size,
        cp_rank=cp_rank,
        aligned_seq_len=aligned_seq_len,
        actual_seq_len=actual_seq_len,
        is_gqa=is_gqa,
    )


def _build_cp_attention_tensors(
    cp_metadata: ContextParallelMetadata,
    seq_len_per_block: int,
    device: Union[str, torch.device],
):
    """Populate device tensors used by CP attention kernels."""
    head_start_global = cp_metadata.kv_len_prev
    head_end_global = (
        cp_metadata.kv_len_prev + cp_metadata.split_list[cp_metadata.cp_rank]
    )
    tail_chunk_id = len(cp_metadata.split_list) - 1 - cp_metadata.cp_rank
    tail_start_global = cp_metadata.kv_len_next
    tail_end_global = tail_start_global + cp_metadata.split_list[tail_chunk_id]

    cp_metadata.kv_len_prev_tensor = torch.tensor(
        cp_metadata.kv_len_prev, device=device, dtype=torch.int32
    )
    cp_metadata.kv_len_next_tensor = torch.tensor(
        cp_metadata.kv_len_next, device=device, dtype=torch.int32
    )
    cp_metadata.actual_seq_q_prev_tensor = torch.tensor(
        cp_metadata.actual_seq_q_prev, device=device, dtype=torch.int32
    )
    cp_metadata.actual_seq_q_next_tensor = torch.tensor(
        cp_metadata.actual_seq_q_next, device=device, dtype=torch.int32
    )

    cp_metadata.attn_mask_seqlens = torch.tensor(
        [[seq_len_per_block], [seq_len_per_block]], device=device, dtype=torch.int32
    )
    cp_metadata.head_attn_nomask_seqlens = torch.tensor(
        [[seq_len_per_block], [head_start_global]], device=device, dtype=torch.int32
    )
    cp_metadata.tail_attn_nomask_seqlens = torch.tensor(
        [[seq_len_per_block], [tail_start_global]], device=device, dtype=torch.int32
    )

    cp_metadata.kv_with_q_head_nomask_idx = torch.arange(
        0, head_start_global, dtype=torch.int32, device=device
    )
    cp_metadata.kv_with_q_head_mask_idx = torch.arange(
        head_start_global, head_end_global, dtype=torch.int32, device=device
    )
    cp_metadata.kv_with_q_tail_nomask_idx = torch.arange(
        0, tail_start_global, dtype=torch.int32, device=device
    )
    cp_metadata.kv_with_q_tail_mask_idx = torch.arange(
        tail_start_global, tail_end_global, dtype=torch.int32, device=device
    )
    return cp_metadata


def calculate_cp_metadata(
    actual_seq_len: int,
    cp_size: int,
    cp_rank: int,
    page_size: int,
    device: Union[str, torch.device],
    is_gqa: bool = True,
):
    """
    Build the canonical CP metadata shared by allocation, transfer, and compute
    """
    (
        cp_block_num,
        aligned_seq_len,
        seq_len_per_block,
        split_list,
        prefix_offsets,
        block_actual_lens,
    ) = _build_cp_block_layout(actual_seq_len, cp_size, page_size)
    cp_metadata = _build_cp_rank_metadata(
        actual_seq_len=actual_seq_len,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_block_num=cp_block_num,
        aligned_seq_len=aligned_seq_len,
        split_list=split_list,
        prefix_offsets=prefix_offsets,
        block_actual_lens=block_actual_lens,
        is_gqa=is_gqa,
    )
    return _build_cp_attention_tensors(cp_metadata, seq_len_per_block, device)


#########################
# Synchronization
#########################

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


def poll_and_all_reduce(pollers, gloo_group: dist.ProcessGroup):
    # at a certain prob, the poll is failed to simulate failure
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_attn_cp_tp_group(
    pollers,
    attn_cp_cpu_group: dist.ProcessGroup,
    attn_tp_cpu_group: dist.ProcessGroup,
):
    # First sync across attn-tp ranks so all TP participants for a given (dp, cp)
    # shard observe the same status transitions.
    polls = poll_and_all_reduce(pollers, attn_tp_cpu_group)

    # Then sync across attn-cp ranks, so all TPxCP participants in one DP shard
    # converge to the same global status.
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(
        tensor_to_reduce,
        op=dist.ReduceOp.MIN,
        group=attn_cp_cpu_group,
    )
    return tensor_to_reduce.tolist()


#########################
# Metadata Buffers
#########################


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self) -> Optional[int]:
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        self.free_slots.append(free_index)


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
    ):
        self.custom_mem_pool = custom_mem_pool
        bootstrap_room_dtype = torch.uint64
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
            # TODO: Fix me when npu backend supports torch.uint64
            bootstrap_room_dtype = torch.int64
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
        elif envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() == "INTRA_NODE_NVLINK":
            device = "cuda"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )
            # Request validation: store bootstrap_room to detect metadata corruption
            self.bootstrap_room = torch.zeros(
                (size, 8), dtype=bootstrap_room_dtype, device=device
            )

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
            self.bootstrap_room.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
            self.bootstrap_room.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
            self.bootstrap_room[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx],
            self.cached_tokens[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
            self.output_topk_p[idx],
            self.output_topk_index[idx],
            self.output_hidden_states[idx],
            self.bootstrap_room[idx],
        )

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )
        # Store bootstrap_room for validation on decode side
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            req.bootstrap_room if req.bootstrap_room is not None else 0
        )


#########################
# Transfer Backend
#########################


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.KVARGS]
) -> Type[KVArgs]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.MANAGER]
) -> Type[CommonKVManager]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.SENDER]
) -> Type[CommonKVSender]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.RECEIVER]
) -> Type[CommonKVReceiver]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.BOOTSTRAP_SERVER]
) -> Type[CommonKVBootstrapServer]: ...


def get_kv_class(
    transfer_backend: TransferBackend, class_type: KVClassType
) -> Optional[Type]:
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.MORI:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mori import (
            MoriKVBootstrapServer,
            MoriKVManager,
            MoriKVReceiver,
            MoriKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MoriKVManager,
            KVClassType.SENDER: MoriKVSender,
            KVClassType.RECEIVER: (MoriKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MoriKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import (
            FakeKVManager,
            FakeKVReceiver,
            FakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: FakeKVManager,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


#########################
# KV Pages
#########################


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaranteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size


def page_indices_to_cp_rank_page_indices(
    page_indices: np.ndarray,
    total_pages: int,
    cp_rank: int,
    cp_size: int,
) -> np.ndarray:
    """
    Filter page_indices (which are *global* page ids in the KV pool) to those
    belonging to the given CP rank for this request.

    For a single request, its pages occupy a contiguous global range
    [first_page, first_page + total_pages). We first compute the local
    split [0, total_pages) across cp_size ranks, then shift that local
    range by first_page back into the global page id space and take
    the intersection with page_indices.

    Returns:
        Subset of page_indices that fall in this rank's global
        [start_page, end_page) slice for the given CP rank.
    """
    if cp_size <= 1:
        return page_indices

    if page_indices.size == 0:
        return np.asarray(page_indices)

    first_page = int(page_indices.min())
    base = total_pages // cp_size
    rem = total_pages % cp_size

    if rem == 0:
        local_start = cp_rank * base
        local_end = local_start + base
    else:
        local_start = cp_rank * base + min(cp_rank, rem)
        n_pages = base + (1 if cp_rank < rem else 0)
        local_end = local_start + n_pages

    # Map back to global page ids.
    start_page = first_page + local_start
    end_page = first_page + local_end

    mask = (page_indices >= start_page) & (page_indices < end_page)
    return np.asarray(page_indices)[mask]


def filter_kv_indices_for_cp_rank(
    kv_mgr: CommonKVManager, kv_indices: np.ndarray, index_slice: slice
) -> Tuple[np.ndarray, slice]:
    """Filters kv_indices and index_slice for the current CP rank."""
    total_pages = len(kv_indices)
    cp_rank = kv_mgr.attn_cp_rank
    cp_size = kv_mgr.attn_cp_size

    rank_page_indices = page_indices_to_cp_rank_page_indices(
        page_indices=kv_indices,
        total_pages=total_pages,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )

    if rank_page_indices.size == 0:
        new_kv_indices = kv_indices[:0]
        new_index_slice = slice(index_slice.start, index_slice.start)
    else:
        mask = np.isin(kv_indices, rank_page_indices)
        if not mask.any():
            new_kv_indices = kv_indices[:0]
            new_index_slice = slice(index_slice.start, index_slice.start)
        else:
            first_pos = int(mask.argmax())
            last_pos = len(mask) - int(mask[::-1].argmax())

            new_kv_indices = kv_indices[first_pos:last_pos]
            new_index_slice = slice(
                index_slice.start + first_pos,
                index_slice.start + last_pos,
            )
    return new_kv_indices, new_index_slice


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, MLATokenToKVPool)


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []
