"""Triton kernel for DeepSeek V4.1 low-ratio indexer score + mask (A5).

A5 variant of the A3 ``low_ratio_index_score`` kernel: identical masking and
candidate-source epilogue, plus in-kernel dequantization of the quantized
A5 index-K pool (fp8_e4m3fn + fp32 scale on arch35; int8 + fp16 scale on
older NPUs), matching ``NPUDeepSeekV4IndexerPool.get_index_k_dequant``.

Replaces the per-request Python loop in
``ascend_dsv4_backend._low_ratio_index_topk_torch_a5`` with one fused kernel:

  score[t, j] = sum_h relu(q[t, h] . (k[slot(t, j)] * scale[slot(t, j)])) * w[t, h]

where ``slot(t, j) = req_to_token[req[t], j * ratio] // ratio``.

Compute structure (prefill rewrite)
-----------------------------------
One program == one (token, position-block) tile, and the whole head lane is a
single Cube matmul followed by a small Vector tail::

    logits = tl.dot(k_tile [BLOCK_N, BLOCK_D], qT [BLOCK_D, BLOCK_H])  # fp32
    score  = sum_h relu(logits[:, h]) * w[h]

The per-slot dequant scale never enters the matmul.  Quantization scales are
amax/448-derived (always >= 0), so::

    relu(s * x) = s * relu(x)        (s >= 0)
    =>  sum_h relu(s_n * dot_h) * w_h  =  s_n * sum_h relu(dot_h) * w_h

the kernel feeds the *raw* fp8/int8 K values to the dot (both are exactly
representable in bf16) and multiplies the scale onto the final [BLOCK_N]
score -- one multiply per output instead of a [BLOCK_N, BLOCK_D] dequant
pass.  This is also what keeps an e4m3 x e4m3 dot available (env
SGLANG_DSV4_INDEX_SCORE_FP8_DOT, arch35 fp8 pools): no mx-style scale
grouping is needed because no scale rides the reduction axis at all.

Addressing: vector offsets stay int32; the int64 part of every row base
(``t * max_lc``, ``r * stride_rtt``, ...) is folded into a scalar pointer
before any vector arithmetic starts.  The wrapper asserts
``k_buf.numel() < 2**31`` so the K-row offsets cannot wrap (the escape hatch
for larger pools is the legacy kernel below).

Masking (fused in-kernel):
  - causal: position j visible iff j < lens[t] = (pos[t] + 1) // ratio,
    invalid positions written as -inf;
  - consume (layers that use candidates): additionally -inf where the
    candidate mask published by the candidate-source layer is False;
  - a tile lying entirely at/past lens[t] short-circuits: it stores -inf
    (and an -inf block score on the candidate-source layer) without
    touching q/k -- max_lc is an upper bound, so short rows hit many of
    these and the MACs they would have run are all dead.

NOT fused (kept in torch on the [T, max_lc] score matrix, per design):
  - the final token-level top-k;
  - the candidate-source layer's block-level top-k (argmax iterations over
    the block scores -- itself a top-k, computed outside the kernel).

For the candidate-source layer the kernel additionally emits per-block amax
scores ``block_scores`` [T, num_blocks] (one program == one candidate block,
i.e. BLOCK_N is forced to the candidate block size): the block amax and the
"newest block forced to +inf" fill are fused in-kernel, so outside the kernel
only the block top-k and the block->position mask expansion remain.

Output: ``out`` [T, max_lc] float32 with -inf at every invalid position --
ready for ``s.topk(...)`` outside.

Numerics vs the torch loop reference
------------------------------------
The reference dequantizes to bf16 (``(k * scale).to(bf16)``) and einsums in
bf16; this kernel dots the exact fp8/int8 K values against bf16 q with an
fp32 accumulator and applies the scale in fp32 afterwards.  That is the same
"one rounding less" deviation the previous fused kernel already carried, in
a different place; top-k picks can differ from the loop reference on
near-ties.  With SGLANG_DSV4_INDEX_SCORE_FP8_DOT=1 q is additionally cast to
e4m3 in-kernel (upstream lightning-indexer semantics); validate top-k overlap
against the bf16 path before enabling it in production.

Environment knobs:
  - SGLANG_DSV4_INDEX_SCORE_BLOCK_N: position tiling (ignored for the
    candidate-source layer, where BLOCK_N == block size).  Default 128.
  - SGLANG_DSV4_INDEX_SCORE_FP8_DOT=1: run the dot in e4m3 x e4m3 (arch35
    fp8 pools only; bf16 dot otherwise).
  - SGLANG_DSV4_INDEX_SCORE_VEC=1: launch the legacy pure-Vector kernel
    (``_index_score_kernel_vec``), the rollback path -- also selected
    automatically when the candidate block size is below tl.dot's 16-wide
    tile minimum.

History: the first fused version looped ``for h in range(H)`` doing
``tl.sum(k * q_h, axis=1)`` per head -- pure Vector-unit code.  msprof at
T=16384 / H=32 / D=128 with a 1M-token req_to_token pool: KERNEL_AIVEC,
650 ms wall, aiv_vec_ratio 0.79, aiv_scalar_ratio 0.37, aiv_mte2_ratio
0.54 -- the QK MACs ran on the Vector fp32 pipe while the Cube (378-432
TFLOPS bf16 / 757-865 TFLOPS fp8 on 950PR) sat idle.  The dot rewrite
moves them to Cube and deletes the per-head q reloads and the int64 vector
index math.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "max_lc", "lc_w", "stride_rtt", "H"])
def _index_score_kernel(
    q_ptr,  # [T, H, D] bf16
    weights_ptr,  # [T, H] bf16
    req_ptr,  # [T] int64 -- request pool index per token
    req_to_token_ptr,  # [R, max_len] int32
    k_buf_ptr,  # [total_slots, D] fp8/int8 (quantized) or bf16 (unquantized)
    k_scale_ptr,  # [total_slots] fp32/fp16 (only when K_QUANTIZED)
    lens_ptr,  # [T] int64 -- visible compressed length per token
    consume_ptr,  # [T, lc_w] bool -- candidate mask (only when HAS_CONSUME)
    out_ptr,  # [T, max_lc] fp32
    block_scores_ptr,  # [T, num_blocks] fp32 (only when IS_CANDIDATE_SOURCE)
    T,
    max_lc,
    lc_w,
    stride_rtt,
    H,
    D,
    RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CONSUME: tl.constexpr,
    IS_CANDIDATE_SOURCE: tl.constexpr,
    K_QUANTIZED: tl.constexpr,
    Q_FP8: tl.constexpr,
):
    t = tl.program_id(0)
    pid_n = tl.program_id(1)
    t64 = t.to(tl.int64)

    # lens 在进比较链之前收窄到 int32（int64 向量比较会在 AIV 上降级成
    # 标量代码）；所有向量下标保持 int32，int64 只出现在下面的标量行基址里。
    lens_t = tl.load(lens_ptr + t).to(tl.int32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = offs_n < max_lc
    out_row = out_ptr + t64 * max_lc

    # 整块落在 lens 之外时不需要任何乘加：max_lc 是上界，短行会命中大量这种
    # tile。直接写 -inf 返回，省掉 gather 和 Cube 乘法（本就是死算）。
    if pid_n * BLOCK_N >= lens_t:
        tl.store(out_row + offs_n, float("-inf"), mask=out_mask)
        if IS_CANDIDATE_SOURCE:
            # 整块不可达 => 不是最新块，amax 恒为 -inf（与 torch 参考一致）。
            tl.store(
                block_scores_ptr + t64 * tl.num_programs(1) + pid_n,
                float("-inf"),
            )
        return

    valid = offs_n < lens_t  # causal: group end passed by this query

    # Full-pool slot of each visible group's base token -> index-pool slot.
    # r 先提升 int64 再乘 stride（req 可能是 int32/int64，大池下 int32 会
    # 溢出）；折进行基址指针后，列偏移保持 int32。masked load 对无效 lane
    # 已经填 0，不需要额外的 tl.where 保护。
    r = tl.load(req_ptr + t).to(tl.int64)
    rtt_row = req_to_token_ptr + r * stride_rtt
    full = tl.load(rtt_row + offs_n * RATIO, mask=valid, other=0)
    if RATIO == 2:
        slots = full >> 1  # full >= 0：移位与 // 等价，且免掉向量整除
    else:
        slots = full

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    # ---- 所有独立 load 一起发出，让 MTE2 在 Cube 计算期间搬运 ----
    # K tile [BLOCK_N, BLOCK_D]：原始量化值（scale 已因式分解到 epilogue，
    # 见模块 docstring）。D=128 时每行恰好 128B，gather 对 L2 友好。
    k_ptrs = k_buf_ptr + slots[:, None] * D + offs_d[None, :]
    k_v = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0)

    # q^T tile [BLOCK_D, BLOCK_H]：d 轴（axis 0）连续，一次 load 覆盖全部
    # 头 —— 旧实现每个头循环迭代都要重新加载一遍 q_h。
    q_row = q_ptr + t64 * (H * D)
    qT = tl.load(
        q_row + offs_d[:, None] + offs_h[None, :] * D,
        mask=d_mask[:, None] & h_mask[None, :],
        other=0.0,
    )

    w = tl.load(weights_ptr + t64 * H + offs_h, mask=h_mask, other=0.0).to(
        tl.float32
    )

    if K_QUANTIZED:
        # 因式分解出来的 per-slot 反量化 scale：只乘最终结果，不进 dot。
        sc = tl.load(k_scale_ptr + slots, mask=valid, other=0.0).to(tl.float32)

    if HAS_CONSUME:
        cm = tl.load(
            consume_ptr + t64 * lc_w + offs_n,
            mask=offs_n < lc_w,
            other=0,
        )

    # ---- Cube：整个头通道一次 matmul，fp32 累加 ----
    if Q_FP8:
        # e4m3 x e4m3 -> fp32：arch35 Cube 上是 bf16 速率的 2 倍。
        logits = tl.dot(k_v, qT.to(k_v.dtype))
    else:
        # fp8/int8/bf16 都能被 bf16 精确表示，转换可被分段搬运进 Cube。
        logits = tl.dot(k_v.to(tl.bfloat16), qT)

    # ---- Vector 尾巴：ReLU -> 头权重 -> 头归约 -> 因式分解的 scale ----
    # relu 在头权重之前（权重可为负，与参考实现一致）。
    score = tl.sum(tl.maximum(logits, 0.0) * w[None, :], axis=1)
    if K_QUANTIZED:
        score = score * sc

    out = tl.where(valid, score, float("-inf"))
    if HAS_CONSUME:
        out = tl.where(valid & (cm != 0), out, float("-inf"))
    tl.store(out_row + offs_n, out, mask=out_mask)

    if IS_CANDIDATE_SOURCE:
        # BLOCK_N == candidate block size here, so this program's tile is
        # exactly one candidate block: emit its amax. Lanes beyond lens are
        # -inf (causal), so an all-unreachable block yields -inf, matching
        # the torch reference's padded amax. The block holding the query's
        # newest position is forced to +inf (always kept), as the reference
        # does with masked_fill before its block top-k.
        bmax = tl.max(out, axis=0)
        is_last = (lens_t > 0) & (pid_n == (lens_t - 1) // BLOCK_N)
        bmax = tl.where(is_last, float("inf"), bmax)
        tl.store(block_scores_ptr + t64 * tl.num_programs(1) + pid_n, bmax)


@triton.jit(do_not_specialize=["T", "max_lc", "lc_w", "stride_rtt", "H"])
def _index_score_kernel_vec(
    q_ptr,  # [T, H, D] bf16
    weights_ptr,  # [T, H] bf16
    req_ptr,  # [T] int -- request pool index per token
    req_to_token_ptr,  # [R, max_len] int32
    k_buf_ptr,  # [total_slots, D] fp8/int8 (quantized) or bf16 (unquantized)
    k_scale_ptr,  # [total_slots] fp32/fp16 (only when K_QUANTIZED)
    lens_ptr,  # [T] int64 -- visible compressed length per token
    consume_ptr,  # [T, lc_w] bool -- candidate mask (only when HAS_CONSUME)
    out_ptr,  # [T, max_lc] fp32
    block_scores_ptr,  # [T, num_blocks] fp32 (only when IS_CANDIDATE_SOURCE)
    T,
    max_lc,
    lc_w,
    stride_rtt,
    H,
    D,
    RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CONSUME: tl.constexpr,
    IS_CANDIDATE_SOURCE: tl.constexpr,
    K_QUANTIZED: tl.constexpr,
):
    """Legacy pure-Vector kernel (per-head mul+reduce loop).

    Kept verbatim as the rollback path (SGLANG_DSV4_INDEX_SCORE_VEC=1) and
    for candidate block sizes below tl.dot's 16-wide tile minimum.  See the
    module docstring for why the dot rewrite replaced it for prefill.
    """
    t = tl.program_id(0)
    pid_n = tl.program_id(1)

    lens_t = tl.load(lens_ptr + t)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + t.to(tl.int64) * max_lc + offs_n
    out_mask = offs_n < max_lc

    # 整块落在 lens 之外时不需要任何乘加：max_lc 是上界，短行会命中大量这种
    # tile。直接写 -inf 返回，省掉 req_to_token gather 和 H 次
    # [BLOCK_N, BLOCK_D] 归约（本就是死算）。
    if pid_n * BLOCK_N >= lens_t:
        tl.store(out_ptrs, float("-inf"), mask=out_mask)
        if IS_CANDIDATE_SOURCE:
            # 整块不可达 => 不是最新块，amax 恒为 -inf（与 torch 参考一致）。
            tl.store(
                block_scores_ptr
                + t.to(tl.int64) * tl.num_programs(1)
                + pid_n,
                float("-inf"),
            )
        return

    valid = offs_n < lens_t  # causal: group end passed by this query

    r = tl.load(req_ptr + t)
    # Full-pool slot of each visible group's base token -> index-pool slot.
    rtt_off = r.to(tl.int64) * stride_rtt + offs_n * RATIO
    rtt_off = tl.where(valid, rtt_off, 0)
    full = tl.load(req_to_token_ptr + rtt_off, mask=valid, other=0)
    slots = (full // RATIO).to(tl.int64)
    slots = tl.where(valid, slots, 0)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    k_ptrs = k_buf_ptr + slots[:, None] * D + offs_d[None, :]
    k_mask = valid[:, None] & d_mask[None, :]
    k_v = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
    if K_QUANTIZED:
        # Per-slot dequant scale, broadcast over D — matches
        # get_index_k_dequant's (k * scale).to(bf16) up to the final rounding,
        # which the fp32 accumulation here skips (slightly more accurate).
        sc = tl.load(k_scale_ptr + slots, mask=valid, other=0.0).to(tl.float32)
        k_v = k_v * sc[:, None]

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for h in range(0, H):
        q_h = tl.load(
            q_ptr + (t.to(tl.int64) * H + h) * D + offs_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        # relu before the head weights, as the reference scores()
        s_h = tl.maximum(tl.sum(k_v * q_h[None, :], axis=1), 0.0)
        w_h = tl.load(weights_ptr + t.to(tl.int64) * H + h).to(tl.float32)
        acc += s_h * w_h

    out = tl.where(valid, acc, float("-inf"))
    if HAS_CONSUME:
        cm = tl.load(
            consume_ptr + t.to(tl.int64) * lc_w + offs_n,
            mask=offs_n < lc_w,
            other=0,
        )
        out = tl.where(valid & (cm != 0), out, float("-inf"))
    tl.store(out_ptrs, out, mask=out_mask)

    if IS_CANDIDATE_SOURCE:
        # BLOCK_N == candidate block size here, so this program's tile is
        # exactly one candidate block: emit its amax. Lanes beyond lens are
        # -inf (causal), so an all-unreachable block yields -inf, matching
        # the torch reference's padded amax. The block holding the query's
        # newest position is forced to +inf (always kept), as the reference
        # does with masked_fill before its block top-k.
        bmax = tl.max(out, axis=0)
        is_last = (lens_t > 0) & (pid_n == (lens_t - 1) // BLOCK_N)
        bmax = tl.where(is_last, float("inf"), bmax)
        tl.store(
            block_scores_ptr + t.to(tl.int64) * tl.num_programs(1) + pid_n, bmax
        )


def low_ratio_index_score_triton(
    q: torch.Tensor,  # [T, H, D] bf16 contiguous
    weights: torch.Tensor,  # [T, H] bf16/fp32 contiguous
    req: torch.Tensor,  # [T] int — request pool index per token
    req_to_token: torch.Tensor,  # [R, max_len] int32
    k_buf: torch.Tensor,  # [total_slots, D] fp8/int8 (quantized) or bf16
    lens: torch.Tensor,  # [T] int64
    max_lc: int,  # host-side upper bound of lens (>= max(lens)); an upper
    # bound is safe: columns beyond a row's lens are -inf-masked in-kernel,
    # so topk's `reach` maps them to -1 exactly like exact-width allocation.
    k_scale: Optional[torch.Tensor] = None,  # [total_slots] fp32/fp16 —
    # required iff k_buf is a quantized (fp8/int8) pool; per-slot dequant
    # scale applied in-kernel.
    consume: Optional[torch.Tensor] = None,  # [T, W] bool
    ratio: int = 2,
    candidate_block_size: Optional[int] = None,  # set on the candidate-source
    # layer: additionally emit [T, cdiv(max_lc, block_size)] fp32 block amax
    # scores (newest block already forced to +inf) for the outside block
    # top-k; BLOCK_N is forced to the block size so one program == one block.
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused indexer scores + dequant + causal (+consume) mask; topk outside.

    Returns ``(s, block_scores)``: ``s`` is [T, max_lc] fp32 with -inf at
    invalid positions; ``block_scores`` is [T, num_blocks] fp32 when
    ``candidate_block_size`` is given, else None. ``max_lc`` must be computed
    on the host (e.g. from seq_lens_cpu) — no device sync here.

    Dispatches to the Cube ``tl.dot`` kernel by default; see the module
    docstring for the SGLANG_DSV4_INDEX_SCORE_* environment knobs (BLOCK_N
    tiling, fp8 dot, legacy Vector rollback).
    """
    T, H, D = q.shape
    if T == 0 or max_lc == 0:
        return q.new_empty(T, 0, dtype=torch.float32), None

    assert q.dtype == torch.bfloat16 and q.is_contiguous()
    assert weights.is_contiguous() and weights.shape == (T, H)
    assert k_buf.is_contiguous() and k_buf.shape[-1] == D
    assert lens.dtype in (torch.int32, torch.int64) and lens.is_contiguous()
    assert ratio in (1, 2)

    if k_scale is not None:
        assert k_buf.dtype in (torch.float8_e4m3fn, torch.int8), (
            "k_scale given but k_buf is not a quantized pool"
        )
        assert k_scale.is_contiguous() and k_scale.dim() == 1
        assert k_scale.shape[0] == k_buf.shape[0]
        k_quantized = True
    else:
        assert k_buf.dtype == torch.bfloat16, (
            "unquantized path expects the bf16 index-K pool"
        )
        k_quantized = False

    if consume is not None:
        # Coerce (not assert): the published mask may be non-bool or a view.
        if consume.dtype != torch.bool:
            consume = consume.to(torch.bool)
        if not consume.is_contiguous():
            consume = consume.contiguous()
        assert consume.shape[0] == T
        lc_w = consume.shape[1]
    else:
        lc_w = max_lc

    use_vec = os.environ.get("SGLANG_DSV4_INDEX_SCORE_VEC", "0") == "1"

    is_candidate_source = candidate_block_size is not None
    if is_candidate_source:
        # tl.arange needs a power of two; one program must cover exactly one
        # candidate block, so the env override is ignored on this layer.
        assert candidate_block_size & (candidate_block_size - 1) == 0, (
            "candidate_block_size must be a power of two for the fused kernel"
        )
        BLOCK_N = candidate_block_size
    else:
        env_bn = os.environ.get("SGLANG_DSV4_INDEX_SCORE_BLOCK_N", "")
        BLOCK_N = int(env_bn) if env_bn else (32 if use_vec else 128)

    num_blocks = triton.cdiv(max_lc, BLOCK_N) if is_candidate_source else 0
    out = torch.empty((T, max_lc), dtype=torch.float32, device=q.device)
    block_scores = (
        torch.empty((T, num_blocks), dtype=torch.float32, device=q.device)
        if is_candidate_source
        else out  # dummy ptr when unused
    )
    grid = (T, triton.cdiv(max_lc, BLOCK_N))

    # tl.dot 要求三个 tile 维都 >= 16：BLOCK_H/BLOCK_D 不足时向上补齐
    # （补零 lane 对结果无贡献）；candidate block 本身小于 16 时无法补
    # （一个 program 必须恰好覆盖一个 block），走 legacy kernel。
    use_dot = (not use_vec) and BLOCK_N >= 16
    if use_dot:
        # int32 K 寻址契约（见模块 docstring）：k_buf 元素偏移必须落在
        # int31 内。更大的池请设 SGLANG_DSV4_INDEX_SCORE_VEC=1 走 int64 老路径。
        assert k_buf.numel() < 2**31, (
            "index-K pool exceeds int32 element offsets; "
            "set SGLANG_DSV4_INDEX_SCORE_VEC=1 for the int64 legacy kernel"
        )
        q_fp8 = (
            os.environ.get("SGLANG_DSV4_INDEX_SCORE_FP8_DOT", "0") == "1"
            and k_quantized
            and k_buf.dtype == torch.float8_e4m3fn
        )
        _index_score_kernel[grid](
            q,
            weights,
            req,
            req_to_token,
            k_buf,
            k_scale if k_scale is not None else k_buf,  # dummy ptr when unused
            lens,
            consume if consume is not None else q,  # dummy ptr when unused
            out,
            block_scores,
            T,
            max_lc,
            lc_w,
            req_to_token.stride(0),
            H,
            D,
            RATIO=ratio,
            BLOCK_N=BLOCK_N,
            BLOCK_H=max(16, triton.next_power_of_2(H)),
            BLOCK_D=max(16, triton.next_power_of_2(D)),
            HAS_CONSUME=consume is not None,
            IS_CANDIDATE_SOURCE=is_candidate_source,
            K_QUANTIZED=k_quantized,
            Q_FP8=q_fp8,
        )
    else:
        _index_score_kernel_vec[grid](
            q,
            weights,
            req,
            req_to_token,
            k_buf,
            k_scale if k_scale is not None else k_buf,  # dummy ptr when unused
            lens,
            consume if consume is not None else q,  # dummy ptr when unused
            out,
            block_scores,
            T,
            max_lc,
            lc_w,
            req_to_token.stride(0),
            H,
            D,
            RATIO=ratio,
            BLOCK_N=BLOCK_N,
            BLOCK_D=triton.next_power_of_2(D),
            HAS_CONSUME=consume is not None,
            IS_CANDIDATE_SOURCE=is_candidate_source,
            K_QUANTIZED=k_quantized,
        )
    return out, block_scores
