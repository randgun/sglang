import logging
import torch
from dataclasses import dataclass
from itertools import accumulate
from typing import Optional, List

from sglang.srt.server_args import get_global_server_args
from sglang.srt.distributed import GroupCoordinator
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)
_cp_diag_log = get_bool_env_var("SGLANG_QWEN3_MOE_DIAG_LOG")


# 新增：KV Gather 辅助函数
def cp_all_gather_kv(local_kv: torch.Tensor, cp_group: GroupCoordinator, stream: torch.cuda.Stream):
    """
    Args:
        local_kv: [local_tokens, num_kv_heads * head_dim] or
                  [local_tokens, num_kv_heads, head_dim]
        cp_group: get_pcp_group() returns a GroupCoordinator; use
                  cp_group.device_group as the real communication group.
    Returns:
        global_kv: [global_tokens, num_kv_heads * head_dim] or
                   [global_tokens, num_kv_heads, head_dim]
    """
    if cp_group is None or cp_group.world_size == 1:
        return local_kv

    assert local_kv.dim() in (
        2,
        3,
    ), f"cp_all_gather_kv expects 2D/3D tensor, got shape={tuple(local_kv.shape)}"

    world_size = cp_group.world_size
    local_kv = local_kv.contiguous()
    if _cp_diag_log:
        logger.info(
            "CP all_gather_kv enter: rank=%s world=%s local_kv=%s",
            getattr(cp_group, "rank_in_group", None),
            world_size,
            tuple(local_kv.shape),
        )

    # CP split can produce different local token counts across ranks.
    # HCCL/NCCL all_gather_into_tensor requires the same count on each rank,
    # so gather lengths first, pad to max_len, and trim after all-gather.
    local_len = torch.tensor([local_kv.shape[0]], dtype=torch.int32, device=local_kv.device)
    gathered_lens = cp_group.all_gather(local_len, dim=0)
    gathered_lens_cpu = gathered_lens.detach().cpu().tolist()

    max_len = max(gathered_lens_cpu)
    if _cp_diag_log:
        logger.info(
            "CP all_gather_kv lens: rank=%s local_len=%d gathered=%s max_len=%d",
            getattr(cp_group, "rank_in_group", None),
            int(local_kv.shape[0]),
            gathered_lens_cpu,
            int(max_len),
        )
    if local_kv.shape[0] != max_len:
        padded_local_kv = torch.zeros(
            (max_len,) + local_kv.shape[1:],
            dtype=local_kv.dtype,
            device=local_kv.device,
        )
        padded_local_kv[: local_kv.shape[0]].copy_(local_kv)
        local_kv = padded_local_kv
        if _cp_diag_log:
            logger.info(
                "CP all_gather_kv padded: rank=%s padded_local_kv=%s",
                getattr(cp_group, "rank_in_group", None),
                tuple(local_kv.shape),
            )

    gather_shape = (max_len * world_size,) + local_kv.shape[1:]
    gathered_padded_kv = torch.empty(gather_shape, dtype=local_kv.dtype, device=local_kv.device)
    cp_group.cp_all_gather_into_tensor_async(gathered_padded_kv, local_kv, stream=stream)
    if len(set(gathered_lens_cpu)) == 1:
        if _cp_diag_log:
            logger.info(
                "CP all_gather_kv exit (uniform): rank=%s gathered=%s",
                getattr(cp_group, "rank_in_group", None),
                tuple(gathered_padded_kv.shape),
            )
        return gathered_padded_kv
    chunks = torch.split(gathered_padded_kv, max_len, dim=0)
    out = torch.cat(
        [chunk[:valid_len] for chunk, valid_len in zip(chunks, gathered_lens_cpu)], dim=0
    )
    if _cp_diag_log:
        logger.info(
            "CP all_gather_kv exit (trim): rank=%s gathered=%s",
            getattr(cp_group, "rank_in_group", None),
            tuple(out.shape),
        )
    return out

    
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
    seq_len_value = int(seq_len)
    seq_len = torch.tensor(seq_len_value)
    bs_per_cp_group = 1
    seq_len_origin = seq_len_value

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
    if _cp_diag_log:
        logger.info(
            "CP meta: seq_len=%d cp_rank=%d cp_size=%d num_heads=%d head_dim=%d "
            "seq_per_batch=%d split_sum=%d max_rank_len=%s per_rank_actual=%s "
            "kv_prev=%d kv_next=%d q_prev=%d q_next=%d",
            int(seq_len_value),
            int(cp_rank),
            int(cp_size),
            int(num_heads),
            int(head_dim),
            int(seq_per_batch),
            int(sum(split_list)),
            max_rank_len,
            per_rank_actual_token,
            int(kv_len_prev),
            int(kv_len_next),
            int(actual_seq_q_prev),
            int(actual_seq_q_next),
        )

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

    # Attention mask 的序列长度信息 [chunk_len, start_global]
    head_attn_nomask_seqlens = torch.tensor(
        [[actual_seq_q_prev], [head_start_global]], dtype=torch.int32, device=device
    )
    tail_attn_nomask_seqlens = torch.tensor(
        [[actual_seq_q_next], [tail_start_global]], dtype=torch.int32, device=device
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
    if _cp_diag_log:
        logger.info(
            "CP meta masks: head_start=%d head_end=%d tail_start=%d tail_end=%d "
            "head_nomask=%d head_mask=%d tail_nomask=%d tail_mask=%d "
            "head_attn_seqlens=%s tail_attn_seqlens=%s attn_mask_seqlens=%s",
            int(head_start_global),
            int(head_end_global),
            int(tail_start_global),
            int(tail_end_global),
            int(kv_with_q_head_nomask_idx_tensor.numel()),
            int(kv_with_q_head_mask_idx_tensor.numel()),
            int(kv_with_q_tail_nomask_idx_tensor.numel()),
            int(kv_with_q_tail_mask_idx_tensor.numel()),
            tuple(head_attn_nomask_seqlens.shape),
            tuple(tail_attn_nomask_seqlens.shape),
            tuple(attn_mask_seqlens_tensor.shape),
        )

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
    out = torch.cat(selected, dim=0)
    if _cp_diag_log:
        logger.info(
            "CP split zigzag: in=%s split_sum=%d splits=%d zigzag=%d out=%s",
            tuple(tensor.shape),
            int(sum(split_list)),
            int(len(split_list)),
            int(len(zigzag_index)),
            tuple(out.shape),
        )
    return out

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
    except_len = sum(reverse_split_len)
    if except_len != tensor.shape[0]:
        if _cp_diag_log:
            logger.info(
                "CP rebuild zigzag skip: expected_len=%d actual_len=%d",
                int(except_len),
                int(tensor.shape[0]),
            )
        return tensor
    if len(cp_reverse_index) != len(reverse_split_len):
        if _cp_diag_log:
            logger.info(
                "CP rebuild zigzag skip: reverse_index_len=%d split_len=%d",
                int(len(cp_reverse_index)),
                int(len(reverse_split_len)),
            )
        return tensor
    tensor_list = list(torch.split(tensor, reverse_split_len, dim=0))
    reordered = [tensor_list[i] for i in cp_reverse_index]
    out = torch.cat(reordered, dim=0)
    if _cp_diag_log:
        logger.info(
            "CP rebuild zigzag: in=%s split_sum=%d splits=%d reverse_idx=%d out=%s",
            tuple(tensor.shape),
            int(sum(reverse_split_len)),
            int(len(reverse_split_len)),
            int(len(cp_reverse_index)),
            tuple(out.shape),
        )
    return out

def is_enable_prefill_cp():
    return get_global_server_args().prefill_context_parallel_size > 1
