import torch
import torch.distributed as dist
from dataclasses import dataclass
from itertools import accumulate
from typing import Optional, List

from sglang.srt.server_args import get_global_server_args
from sglang.srt.distributed import GroupCoordinator


# 新增：KV Gather 辅助函数
def cp_all_gather_kv(local_kv: torch.Tensor, cp_group: GroupCoordinator):
    """
    Args:
        local_kv: [local_tokens, num_kv_heads * head_dim] or
                  [local_tokens, num_kv_heads, head_dim]
    Returns:
        global_kv: [global_tokens, num_kv_heads * head_dim] or
                   [global_tokens, num_kv_heads, head_dim]
    """
    if cp_group is None or dist.get_world_size(group=cp_group) == 1:
        return local_kv
    assert local_kv.dim() in (
        2,
        3,
    ), f"cp_all_gather_kv expects 2D/3D tensor, got shape={tuple(local_kv.shape)}"

    world_size = cp_group.size
    output_list = [torch.empty_like(local_kv) for _ in range(world_size)]
    torch.distributed.all_gather(output_list, local_kv, group=cp_group)
    # For RadixAttention path, token/sequence is the leading dimension.
    # Concatenate along dim=0 to rebuild global KV tokens.
    global_kv = torch.cat(output_list, dim=0)
    assert (
        global_kv.shape[1:] == local_kv.shape[1:]
    ), f"KV head dims changed after all_gather: local={tuple(local_kv.shape)} global={tuple(global_kv.shape)}"

    return global_kv

def gqa_use_prefill_cp(forward_batch, gqa_enable_prefill_cp=None):
    if gqa_enable_prefill_cp is None:
        gqa_enable_prefill_cp = is_enable_prefill_cp()
    if (
        forward_batch.gqa_cp_metadata is not None
        and gqa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    ):
        return True
    else:
        return False


@dataclass
class ContextParallelMetadata:
    split_list: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    zigzag_index: Optional[List[int]] = None
    per_rank_actual_token: Optional[List[int]] = None
    reverse_split_len: Optional[List[int]] = None
    cp_reverse_index: Optional[List[int]] = None
    kv_len_prev: int = -1
    kv_len_next: int = -1
    actual_seq_q_prev: int = -1
    actual_seq_q_next: int = -1
    q_head_num: Optional[int] = None  # Query头数
    kv_head_num: Optional[int] = None  # Key/Value头数
    head_dim: Optional[int] = None  # 头维度
    kv_len_prev_tensor: Optional[torch.Tensor] = None
    kv_len_next_tensor: Optional[torch.Tensor] = None
    actual_seq_q_prev_tensor: Optional[torch.Tensor] = None
    actual_seq_q_next_tensor: Optional[torch.Tensor] = None
    total_seq_lens: Optional[int] = None
    kv_with_q_head_nomask_idx: Optional[torch.Tensor] = None
    kv_with_q_head_mask_idx: Optional[torch.Tensor] = None
    kv_with_q_tail_nomask_idx: Optional[torch.Tensor] = None
    kv_with_q_tail_mask_idx: Optional[torch.Tensor] = None
    head_attn_nomask_seqlens: Optional[torch.Tensor] = None
    tail_attn_nomask_seqlens: Optional[torch.Tensor] = None
    attn_mask_seqlens: Optional[torch.Tensor] = None


def prepare_qwen_cp_metadata(
    seq_len: int,
    cp_rank: int,
    cp_size: int,
    num_heads: int,
    head_dim: int,
):
    """Prepare context parallelism metadata for QWEN models using zigzag strategy.

    This function implements the zigzag segmentation strategy to balance
    computational load across CP ranks during prefill phase.

    Args:
        seq_len: Total sequence length for the request
        cp_rank: Current context parallel rank
        cp_size: Total number of CP ranks
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head

    Returns:
        ContextParallelMetadata: Metadata for CP coordination
    """
    if cp_size <= 1:
        return None
    # kv_len = torch.tensor(kv_len)
    # bs_per_cp_group = 1
    # kv_len_origin = kv_len
    seq_len = torch.tensor(seq_len)
    bs_per_cp_group = 1
    seq_len_origin = seq_len

    cp_segment_num = cp_size * 2
    seq_per_batch = seq_len // cp_segment_num
    split_list = seq_per_batch.repeat_interleave(cp_segment_num).int().tolist()

    remainder = seq_len % cp_segment_num
    if remainder > 0:
        split_list[:remainder] = [x + 1 for x in split_list[:remainder]]

    seq_max_rank_len = (seq_len + cp_size - 1) // cp_size
    max_rank_len = seq_max_rank_len.repeat_interleave(cp_size).int().tolist()

    zigzag_index = list(
        range(cp_rank, cp_rank + bs_per_cp_group * cp_segment_num, cp_segment_num)
    ) + list(
        range(
            cp_segment_num - cp_rank - 1,
            bs_per_cp_group * cp_segment_num,
            cp_segment_num,
        )
    )

    per_rank_actual_token = list(
        split_list[i] + split_list[cp_size * 2 - i - 1] for i in range(cp_size)
    )

    reverse_split_len = [
        element
        for i in range(cp_size)
        for element in (split_list[i], split_list[cp_size * 2 - i - 1])
    ]

    cp_reverse_index = []
    for batch_id in range(bs_per_cp_group):
        cp_reverse_index.extend(
            list(range(batch_id, cp_segment_num * bs_per_cp_group, 2 * bs_per_cp_group))
            + list(
                range(
                    (cp_segment_num - 1) * bs_per_cp_group + batch_id,
                    0,
                    -2 * bs_per_cp_group,
                )
            )
        )

    prefix_sum_list = list(accumulate(split_list))

    kv_len_prev = prefix_sum_list[cp_rank]
    kv_len_next = prefix_sum_list[cp_size * 2 - cp_rank - 1]
    actual_seq_q_prev = split_list[cp_rank]
    actual_seq_q_next = split_list[cp_size * 2 - cp_rank - 1]

    device = "npu"

    kv_len_prev_tensor = torch.tensor(kv_len_prev, device=device, dtype=torch.int32)
    kv_len_next_tensor = torch.tensor(kv_len_next, device=device, dtype=torch.int32)
    actual_seq_q_prev_tensor = torch.tensor(actual_seq_q_prev, device=device, dtype=torch.int32)
    actual_seq_q_next_tensor = torch.tensor(actual_seq_q_next, device=device, dtype=torch.int32)

    # ========== Mask 计算相关元数据 ==========
    # 计算 prefix offsets 用于定位每个 chunk 的全局起始位置
    prefix_offsets = [0] + prefix_sum_list

    # 当前 rank 对应的 head chunk 和 tail chunk 的索引
    head_chunk_id = cp_rank
    tail_chunk_id = cp_segment_num - 1 - cp_rank

    # Head chunk 的全局起始和结束位置
    head_start_global = prefix_offsets[head_chunk_id]
    head_end_global = prefix_offsets[head_chunk_id + 1]
    # Tail chunk 的全局起始和结束位置
    tail_start_global = prefix_offsets[tail_chunk_id]
    tail_end_global = prefix_offsets[tail_chunk_id + 1]

    # Attention mask 的序列长度信息 [seq_per_batch, start_global]
    head_attn_nomask_seqlens = torch.tensor(
        [[seq_per_batch], [head_start_global]], dtype=torch.int32, device=device
    )
    tail_attn_nomask_seqlens = torch.tensor(
        [[seq_per_batch], [tail_start_global]], dtype=torch.int32, device=device
    )

    # 包含 Head 和 Tail 两个 block 的长度，供算子 batch 处理使用
    attn_mask_seqlens_tensor = torch.tensor(
        [[actual_seq_q_prev], [actual_seq_q_next]], dtype=torch.int32, device=device
    )

    # KV 与 Q 的 mask 索引计算
    # Head Chunk: [0, start) 是 nomask, [start, end) 是 causal mask
    kv_with_q_head_nomask_idx = list(range(0, head_start_global))
    kv_with_q_head_mask_idx = list(range(head_start_global, head_end_global))
    # Tail Chunk: [0, start) 是 nomask, [start, end) 是 causal mask
    kv_with_q_tail_nomask_idx = list(range(0, tail_start_global))
    kv_with_q_tail_mask_idx = list(range(tail_start_global, tail_end_global))

    kv_with_q_head_nomask_idx_tensor = torch.tensor(
        kv_with_q_head_nomask_idx, dtype=torch.int32, device=device
    ) if kv_with_q_head_nomask_idx else torch.empty(0, dtype=torch.int32, device=device)
    kv_with_q_head_mask_idx_tensor = torch.tensor(
        kv_with_q_head_mask_idx, dtype=torch.int32, device=device
    ) if kv_with_q_head_mask_idx else torch.empty(0, dtype=torch.int32, device=device)
    kv_with_q_tail_nomask_idx_tensor = torch.tensor(
        kv_with_q_tail_nomask_idx, dtype=torch.int32, device=device
    ) if kv_with_q_tail_nomask_idx else torch.empty(0, dtype=torch.int32, device=device)
    kv_with_q_tail_mask_idx_tensor = torch.tensor(
        kv_with_q_tail_mask_idx, dtype=torch.int32, device=device
    ) if kv_with_q_tail_mask_idx else torch.empty(0, dtype=torch.int32, device=device)

    metadata = ContextParallelMetadata(
        split_list=split_list,
        max_rank_len=max_rank_len,
        zigzag_index=zigzag_index,
        per_rank_actual_token=per_rank_actual_token,
        reverse_split_len=reverse_split_len,
        cp_reverse_index=cp_reverse_index,
        kv_len_prev=kv_len_prev,
        kv_len_next=kv_len_next,
        actual_seq_q_prev=actual_seq_q_prev,
        actual_seq_q_next=actual_seq_q_next,
        kv_len_prev_tensor=kv_len_prev_tensor,
        kv_len_next_tensor=kv_len_next_tensor,
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,
        total_seq_lens=seq_len_origin,
        q_head_num=num_heads,
        head_dim=head_dim,
        # Mask 相关元数据
        kv_with_q_head_nomask_idx=kv_with_q_head_nomask_idx_tensor,
        kv_with_q_head_mask_idx=kv_with_q_head_mask_idx_tensor,
        kv_with_q_tail_nomask_idx=kv_with_q_tail_nomask_idx_tensor,
        kv_with_q_tail_mask_idx=kv_with_q_tail_mask_idx_tensor,
        head_attn_nomask_seqlens=head_attn_nomask_seqlens,
        tail_attn_nomask_seqlens=tail_attn_nomask_seqlens,
        attn_mask_seqlens=attn_mask_seqlens_tensor,
    )

    return metadata


def cp_split_tensor_by_zigzag(
    tensor: torch.Tensor,
    split_list: List[int],
    zigzag_index: List[int],
):
    """Split tensor according to CP zigzag strategy.

    Args:
        tensor: Input tensor to split
        split_list: List of split sizes
        zigzag_index: Zigzag reordering index

    Returns:
        torch.Tensor: Split tensor for current CP rank
    """
    tensor_list = list(torch.split(tensor, split_list, dim=0))
    selected = [tensor_list[i] for i in zigzag_index]
    return torch.cat(selected, dim=0)

def cp_rebuild_tensor_by_zigzag(
    tensor: torch.Tensor,
    reverse_split_len: List[int],
    cp_reverse_index: List[int],
):
    """Rebuild tensor from CP output back to original order.

    Args:
        tensor: Output tensor from CP computation
        reverse_split_len: List of split sizes in reverse order
        cp_reverse_index: Reverse index for reordering

    Returns:
        torch.Tensor: Rebuilt tensor in original token order
    """
    tensor_list = list(torch.split(tensor, reverse_split_len, dim=0))
    reordered = [tensor_list[i] for i in cp_reverse_index]
    return torch.cat(reordered, dim=0)

def is_enable_prefill_cp():
    return get_global_server_args().prefill_context_parallel_size > 1

