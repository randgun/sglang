from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Iterable, List, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope, fused_rope_inplace
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    BuildStepLocal,
    CommitKvProj,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v4 import (
    DEEPSEEK_V4_STACKED_PARAMS_MAPPING,
    DeepseekV4DecoderLayer,
    MqaAttentionBase,
    _dequant_fp8_wo_a,
    hc_head_torch,
    make_hc_head_params,
)
from sglang.srt.models.dspark import (
    DSparkConfidenceHead,
    StepSampler,
    gather_and_crop_vocab,
    run_markov_block,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.utils import add_prefix, is_blackwell_supported, is_npu
from sglang.srt.utils.invariants import Bucket, InClosedRange, Invariant, expect
from sglang.srt.utils.async_probe import maybe_detect_in_closed_range

logger = logging.getLogger(__name__)

_PAD_NUM_HEADS = 64

# DSpark confidence is a per-token score that must stay in [0, 1].
_CONFIDENCE = Invariant(
    "dspark.model.confidence", Bucket.GUARD, InClosedRange(0.0, 1.0)
)
_is_npu = is_npu()


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    if _is_npu:
        import torch_npu

        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            _get_contig_freqs_real_imag,
        )

        # aclnnIndex does not support complex64, so split the full frequency
        # table into contiguous real/imag tensors before indexing it.
        freqs_real, freqs_imag = _get_contig_freqs_real_imag(freqs_cis)
        cos_half = freqs_real[positions].to(x.dtype)
        sin_half = freqs_imag[positions].to(x.dtype)
        if inverse:
            sin_half = -sin_half

        rope_dim = x.shape[-1]
        cos = (
            cos_half.repeat_interleave(2, dim=-1)
            .view(-1, 1, 1, rope_dim)
            .contiguous()
        )
        sin = (
            sin_half.repeat_interleave(2, dim=-1)
            .view(-1, 1, 1, rope_dim)
            .contiguous()
        )
        x_3d = x.reshape(x.shape[0], -1, rope_dim)
        rotated = torch_npu.npu_rotary_mul(
            x_3d.unsqueeze(1), cos, sin, rotary_mode="interleave"
        )
        x.copy_(rotated.squeeze(1).view_as(x))
        return x

    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    freqs_cis = freqs_cis[positions]
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(x.size(0), 1, x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


class DSparkAttention(MqaAttentionBase):

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            attn_tp_rank=get_parallel().attn_tp_rank,
            attn_tp_size=get_parallel().attn_tp_size,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_a_fp8=False,
            wo_a_keeps_quant_config=False,
            wo_b_reduce_results=True,
            rope_original_seq_len=0,
        )
        assert (
            self.compress_ratio == 0
        ), "DSpark draft attention requires compress_ratio == 0."
        self.window_size = int(
            getattr(config, "sliding_window", None) or config.window_size
        )

        self.attn = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self._use_fast_kernel = envs.SGLANG_DSPARK_FAST_KERNEL.get()
        self.alt_streams = alt_streams
        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64
        self._debug_probe_key: Optional[str] = None
        self._debug_probe_count = 0

    def set_debug_probe_context(self, probe_key: Optional[str]) -> None:
        self._debug_probe_key = probe_key

    def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:
        kv, _ = self.wkv(x)
        return kv

    def _store_block_kv(
        self,
        *,
        kv: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )

    def _compute_q(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self._use_fast_kernel:
            if q_out is None:
                q_out = torch.empty_like(q)
            fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
            return q_out
        else:
            q = q * torch.rsqrt(
                q.float().square().mean(-1, keepdim=True) + self.eps
            ).to(q.dtype)
            apply_rotary_emb(
                q[..., -self.rope_head_dim :], self.freqs_cis, positions
            )
            if q_out is not None:
                q_out.copy_(q)
                return q_out
            return q

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        pool = _resolve_dspark_pool()
        attn_backend = get_attn_backend()
        rd = self.rope_head_dim

        enable_multi_stream = (
            self.alt_streams is not None
            and get_is_capture_mode()
            and hidden_states.shape[0] <= self._multi_stream_bs_limit
        )

        q_padded: Optional[torch.Tensor] = None
        q_out: Optional[torch.Tensor] = None
        if self.n_local_heads < _PAD_NUM_HEADS:
            q_padded = (
                hidden_states.new_zeros(
                    hidden_states.shape[0], _PAD_NUM_HEADS, self.head_dim
                )
                if _is_npu
                else hidden_states.new_empty(
                    hidden_states.shape[0], _PAD_NUM_HEADS, self.head_dim
                )
            )
            q_out = q_padded[:, : self.n_local_heads, :]

        if enable_multi_stream:
            current_stream = torch.cuda.current_stream()
            stream_kv = self.alt_streams[0]
            stream_kv.wait_stream(current_stream)
            with torch.cuda.stream(stream_kv):
                kv = self.kv_proj_only(hidden_states)
                self._store_block_kv(
                    kv=kv,
                    positions=positions,
                    forward_batch=forward_batch,
                    attn_backend=attn_backend,
                    pool=pool,
                )
            q = self._compute_q(hidden_states, positions, q_out=q_out)
            current_stream.wait_stream(stream_kv)
        else:
            kv = self.kv_proj_only(hidden_states)
            self._store_block_kv(
                kv=kv,
                positions=positions,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
                pool=pool,
            )
            q = self._compute_q(hidden_states, positions, q_out=q_out)

        if q_padded is not None:
            q = q_padded
        attn_sink = self._local_attn_sink()

        do_probe = (
            self._debug_probe_key is not None
            and self._debug_probe_key.startswith(
                ("dspark-kv-debug", "dspark-token-debug")
            )
            and get_parallel().tp_rank == 0
            and self._debug_probe_count < 3
        )
        if do_probe:
            fm = getattr(attn_backend, "forward_metadata", None)
            page_table = getattr(fm, "swa_page_table", None)
            actual_kv = getattr(fm, "actual_seq_lengths_kv", None)
            actual_q_pa = getattr(fm, "actual_seq_lengths_q_pa", None)
            swa_out = attn_backend.get_swa_out_cache_loc(forward_batch)
            logger.warning(
                "DSpark attention metadata: rid=%s step=%d stage=%d "
                "positions=%s seq_lens=%s actual_kv=%s actual_q_pa=%s "
                "page_table_head=%s out_cache_loc=%s swa_out=%s "
                "q=(shape=%s,abs_mean=%.8f,rms=%.8f) "
                "kv=(shape=%s,abs_mean=%.8f,rms=%.8f)",
                self._debug_probe_key,
                self._debug_probe_count,
                self.layer_id,
                positions.detach().cpu().tolist(),
                forward_batch.seq_lens.detach().cpu().tolist(),
                actual_kv.detach().cpu().tolist() if actual_kv is not None else None,
                (
                    actual_q_pa.detach().cpu().tolist()
                    if actual_q_pa is not None
                    else None
                ),
                (
                    page_table[0, :8].detach().cpu().tolist()
                    if page_table is not None and page_table.numel() > 0
                    else None
                ),
                (
                    forward_batch.out_cache_loc.detach().cpu().tolist()
                    if forward_batch.out_cache_loc is not None
                    else None
                ),
                swa_out.detach().cpu().tolist(),
                tuple(q.shape),
                float(q.detach().float().abs().mean().item()),
                float(torch.sqrt(q.detach().float().square().mean()).item()),
                tuple(kv.shape),
                float(kv.detach().float().abs().mean().item()),
                float(torch.sqrt(kv.detach().float().square().mean()).item()),
            )

        o = attn_backend.forward(
            q=q,
            k=kv,
            v=kv,
            layer=self.attn,
            forward_batch=forward_batch,
            compress_ratio=0,
            attn_sink=attn_sink,
            save_kv_cache=False,
        )

        history_ablation = None
        if do_probe and hasattr(pool, "get_swa_buffer"):
            prefix_len = int(forward_batch.seq_lens[0].detach().cpu().item())
            req_idx = int(
                forward_batch.req_pool_indices[0].detach().cpu().item()
            )
            full_history = attn_backend.req_to_token_pool.req_to_token[
                req_idx, :prefix_len
            ]
            swa_history = pool.translate_loc_from_full_to_swa(
                full_history
            ).to(torch.int64)
            cache = pool.get_swa_buffer(self.layer_id).flatten(0, 1)
            saved_history = cache.index_select(0, swa_history).clone()
            try:
                cache.index_fill_(0, swa_history, 0)
                o_without_history = attn_backend.forward(
                    q=q,
                    k=kv,
                    v=kv,
                    layer=self.attn,
                    forward_batch=forward_batch,
                    compress_ratio=0,
                    attn_sink=attn_sink,
                    save_kv_cache=False,
                )
            finally:
                cache.index_copy_(0, swa_history, saved_history)
            actual = o.detach().float().cpu()
            ablated = o_without_history.detach().float().cpu()
            delta = actual - ablated
            history_ablation = {
                "history_swa": swa_history.detach().cpu().tolist(),
                "out_abs_mean": round(float(actual.abs().mean().item()), 8),
                "out_rms": round(
                    float(torch.sqrt(actual.square().mean()).item()), 8
                ),
                "delta_abs_mean": round(
                    float(delta.abs().mean().item()), 8
                ),
                "delta_max": round(float(delta.abs().max().item()), 8),
                "relative_rms": round(
                    float(
                        torch.sqrt(delta.square().mean())
                        / torch.sqrt(actual.square().mean()).clamp_min(1e-12)
                    ),
                    8,
                ),
                "cosine": round(
                    float(
                        F.cosine_similarity(
                            actual.reshape(1, -1),
                            ablated.reshape(1, -1),
                            dim=-1,
                        ).item()
                    ),
                    10,
                ),
            }
            logger.warning(
                "DSpark attention history ablation: rid=%s step=%d stage=%d %s",
                self._debug_probe_key,
                self._debug_probe_count,
                self.layer_id,
                history_ablation,
            )

        if o.shape[1] != self.n_local_heads:
            o = o[:, : self.n_local_heads, :]

        if self._use_fast_kernel:
            fused_rope_inplace(
                o[..., -rd:], None, self.freqs_cis, positions=positions, inverse=True
            )
        else:
            apply_rotary_emb(
                o[..., -rd:], self.freqs_cis, positions, inverse=True
            )

        o = o.view(
            o.shape[0],
            self.n_local_groups,
            o.shape[1] * o.shape[2] // self.n_local_groups,
        )
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if self._use_fast_kernel:
            o = torch.einsum("bgd,grd->bgr", o, wo_a)
        else:
            o = torch.einsum("bgd,grd->bgr", o.float(), wo_a.float()).to(q.dtype)
        out, _ = self.wo_b(o.reshape(o.shape[0], o.shape[1] * o.shape[2]))
        if do_probe:
            out_float = out.detach().float()
            logger.warning(
                "DSpark attention output: rid=%s step=%d stage=%d "
                "out=(shape=%s,abs_mean=%.8f,rms=%.8f,abs_max=%.8f)",
                self._debug_probe_key,
                self._debug_probe_count,
                self.layer_id,
                tuple(out.shape),
                float(out_float.abs().mean().item()),
                float(torch.sqrt(out_float.square().mean()).item()),
                float(out_float.abs().max().item()),
            )
            self._debug_probe_count += 1
        return out


def _resolve_dspark_pool() -> DeepSeekV4TokenToKVPool:
    pool = get_token_to_kv_pool()
    assert isinstance(pool, DeepSeekV4TokenToKVPool), (
        "DSpark draft attention requires a DeepSeekV4TokenToKVPool, "
        f"got {type(pool).__name__}."
    )
    return pool


class MarkovW2ShardGeometry(msgspec.Struct, frozen=True):

    tp_size: int
    org_vocab_start: int
    org_vocab_end: int
    num_embeddings_per_partition: int
    num_embeddings_padded: int


class DSparkV4MarkovHead(nn.Module):

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"DSparkV4MarkovHead requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = VocabParallelEmbedding(
            self.vocab_size, self.markov_rank, enable_tp=False
        )
        self._opt_markov_w2_bf16 = envs.SGLANG_DSPARK_OPT_MARKOV_W2_BF16.get()
        self._opt_markov_w2_tp_shard = envs.SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD.get()
        markov_w2_dtype = torch.bfloat16 if self._opt_markov_w2_bf16 else torch.float32
        self.markov_w2 = nn.Linear(
            self.markov_rank, self.vocab_size, bias=False, dtype=markov_w2_dtype
        )
        self._tp_shard: Optional[MarkovW2ShardGeometry] = None

    def configure_tp_shard(self, *, lm_head: nn.Module) -> None:
        if not self._opt_markov_w2_tp_shard:
            return
        if int(lm_head.org_vocab_size) != self.vocab_size:
            raise ValueError(
                "DSpark markov_w2 TP-shard requires lm_head.org_vocab_size == "
                f"markov vocab_size, got {int(lm_head.org_vocab_size)} vs "
                f"{self.vocab_size}."
            )
        tp_size = int(lm_head.tp_size)
        per_partition = int(lm_head.num_embeddings_per_partition)
        num_padded = int(lm_head.num_embeddings_padded)
        if per_partition * tp_size != num_padded:
            raise ValueError(
                "DSpark markov_w2 TP-shard could not align to the lm_head partition: "
                f"num_embeddings_per_partition({per_partition}) * tp_size({tp_size}) != "
                f"num_embeddings_padded({num_padded})."
            )
        attn_tp_size = get_parallel().attn_tp_group.world_size
        if attn_tp_size != tp_size:
            raise ValueError(
                "DSpark markov_w2 TP-shard needs the attn-TP group (used for the per-step "
                f"all-gather) to equal the lm_head shard group, got attn_tp_size="
                f"{attn_tp_size} vs lm_head tp_size={tp_size}. This config (e.g. DP "
                "attention without --enable-dp-lm-head, where lm_head shards over the "
                "global TP group) is unsupported; disable "
                "SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD."
            )
        self._tp_shard = MarkovW2ShardGeometry(
            tp_size=tp_size,
            org_vocab_start=int(lm_head.shard_indices.org_vocab_start_index),
            org_vocab_end=int(lm_head.shard_indices.org_vocab_end_index),
            num_embeddings_per_partition=per_partition,
            num_embeddings_padded=num_padded,
        )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(
        self, latent_states: torch.Tensor, *, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        weight = self.markov_w2.weight if weight is None else weight
        if self._opt_markov_w2_bf16:
            return F.linear(latent_states.to(weight.dtype), weight).float()
        return F.linear(latent_states.float(), weight)

    def compute_step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._tp_shard is not None:
            return self._apply_step_logits_sharded(
                base_local=logits, token_ids=token_ids
            )
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def _apply_step_logits_sharded(
        self, *, base_local: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        shard = self._tp_shard
        latent = self.get_prev_embeddings(token_ids)
        weight_local = self.markov_w2.weight[
            shard.org_vocab_start : shard.org_vocab_end
        ]
        if self._opt_markov_w2_bf16:
            bias = F.linear(latent.to(weight_local.dtype), weight_local)
        else:
            bias = F.linear(latent.float(), weight_local)
        step_local = BuildStepLocal.execute(bias=bias, base_local=base_local)
        if shard.tp_size > 1:
            full = get_parallel().attn_tp_group.all_gather(step_local, dim=-1)
        else:
            full = step_local
        return full[..., : self.vocab_size]

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.get_prev_embeddings(token_ids)
        logits = self.project_bias(embed)
        return logits, embed

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )


def build_dspark_v4_confidence_head(
    *, config: DeepSeekV4Config, markov_rank: int
) -> Optional[DSparkConfidenceHead]:
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    if not hasattr(config, "enable_confidence_head"):
        logger.warning(
            "DSpark draft config has no enable_confidence_head field; treating the "
            "confidence head as enabled."
        )
    with_markov_cfg = getattr(config, "confidence_head_with_markov", None)
    with_markov = (
        (markov_rank > 0) if with_markov_cfg is None else bool(with_markov_cfg)
    )
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark V4 confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=int(config.hidden_size),
        markov_rank=int(markov_rank),
        with_markov=with_markov,
        bias=False,
    )


class DSparkV4Stage(DeepseekV4DecoderLayer):

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        stage_id: int,
        num_stages: int,
        num_target_layers: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            is_nextn=True,
            alt_streams=alt_streams,
        )
        self.stage_id = stage_id
        self.dim = config.hidden_size

        if stage_id == 0:
            if num_target_layers <= 0:
                raise ValueError(
                    "DSpark needs target layers for the target-hidden projection."
                )
            self.main_proj = ReplicatedLinear(
                config.hidden_size * num_target_layers,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("main_proj", prefix),
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if stage_id == num_stages - 1:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            (
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
            ) = make_hc_head_params(config.hc_mult, config.hidden_size)

    def _build_self_attn(
        self,
        *,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[List[torch.cuda.Stream]],
        compress_ratio_override: Optional[int],
    ) -> nn.Module:
        del compress_ratio_override
        return DSparkAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
        )

    def _hc_pre_block(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, post, comb, _ = self.hc_pre(x, hc_fn, hc_scale, hc_base)
        return y, post, comb

    def _hc_post_block(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return self.hc_post(x, residual, post, comb)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        x, post, comb = self._hc_pre_block(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
            x = self.self_attn(positions, x, forward_batch)
        x = self._hc_post_block(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre_block(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self._run_ffn(x, forward_batch)
        x = self._hc_post_block(x, residual, post, comb)
        return x

    def _run_ffn(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        input_ids = forward_batch.input_ids
        if input_ids is None:
            raise RuntimeError(
                "DeepSeek-V4 DSpark MoE requires forward_batch.input_ids for "
                "hash routing and TP-attention/A2A token scatter."
            )
        y = self._run_moe_ffn_dp_sync(
            x,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids,
        )
        return y.view(shape)


class DeepseekV4ForCausalLMDSpark(nn.Module):
    # DeepSeek-V4 DSpark checkpoints carry QuaRot-aligned, MTP-local
    # embedding/head weights. They must not be replaced by the target model's
    # vocabulary modules.
    uses_own_vocab_modules = True

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark V4 draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(
            dspark_config.resolve_gamma(default=int(config.num_hidden_layers))
        )
        self.block_size = self.gamma
        if dspark_config.target_layer_ids is not None:
            self.num_stages = len(dspark_config.target_layer_ids)
        else:
            self.num_stages = int(getattr(config, "num_nextn_predict_layers", 1) or 1)

        target_num_layers = (
            int(dspark_config.num_target_layers)
            if dspark_config.num_target_layers is not None
            else int(getattr(config, "num_hidden_layers", 1))
        )
        if dspark_config.target_layer_ids is not None:
            self.num_target_features = len(dspark_config.target_layer_ids)
        else:
            self.num_target_features = target_num_layers

        self.start_layer = 0
        self.end_layer = self.num_stages
        use_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and envs.SGLANG_DSPARK_ENABLE_MULTI_STREAM.get()
            and torch.cuda.is_available()
        )
        self.alt_streams: Optional[List[torch.cuda.Stream]] = (
            [torch.cuda.Stream()] if use_multi_stream else None
        )
        self.stages = nn.ModuleList(
            [
                DSparkV4Stage(
                    config=config,
                    layer_id=stage_id,
                    stage_id=stage_id,
                    num_stages=self.num_stages,
                    num_target_layers=self.num_target_features,
                    quant_config=quant_config,
                    prefix=add_prefix(f"stages.{stage_id}", prefix),
                    alt_streams=self.alt_streams,
                )
                for stage_id in range(self.num_stages)
            ]
        )
        self.markov_head = DSparkV4MarkovHead(
            vocab_size=int(config.vocab_size),
            markov_rank=int(dspark_config.markov_rank),
        )
        self.confidence_head = build_dspark_v4_confidence_head(
            config=config, markov_rank=int(dspark_config.markov_rank)
        )
        self.hc_mult = int(config.hc_mult)
        self.norm_eps = float(config.rms_norm_eps)
        self.hc_eps = float(config.hc_eps)

        # These weights are FLOAT in the ModelSlim checkpoint even when the
        # stage MoE/linear weights are quantized, so intentionally do not pass
        # the draft quant_config to either vocabulary module.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
            enable_tp=not is_dp_attention_enabled(),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_server_args().enable_dp_lm_head,
        )
        self._use_fp32_lm_head = envs.SGLANG_DSPARK_FP32_LM_HEAD.get()
        self._opt_markov_w2_tp_shard = envs.SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD.get()
        self.markov_head.configure_tp_shard(lm_head=self.lm_head)

    @property
    def enable_confidence_head(self) -> bool:
        return self.confidence_head is not None

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        # Kept for API compatibility with generic DSpark worker code. This
        # model owns QuaRot-aligned MTP vocabulary modules, so target modules
        # must never overwrite them.
        del embed_tokens, lm_head
        self.markov_head.configure_tp_shard(lm_head=self.lm_head)

    def set_attention_probe_context(self, probe_key: Optional[str]) -> None:
        for stage in self.stages:
            stage.self_attn.set_debug_probe_context(probe_key)

    def project_target_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        stage0 = self.stages[0]
        projected, _ = stage0.main_proj(main_hidden)
        return stage0.main_norm(projected)

    def write_target_hidden_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
        probe_key: Optional[str] = None,
        probe_phase: str = "unknown",
    ) -> None:
        probe_count = getattr(self, "_kv_pipeline_probe_counts", {}).get(
            probe_key, 0
        )
        do_probe = (
            probe_key is not None
            and probe_key.startswith(("dspark-kv-debug", "dspark-token-debug"))
            and get_parallel().tp_rank == 0
            and probe_count < 3
        )

        main_x = self.project_target_hidden(main_hidden)
        swa_loc = swa_loc.to(torch.int32)
        kvs = CommitKvProj.execute(
            main_x=main_x,
            wkv_linears=[stage.self_attn.wkv for stage in self.stages],
        )

        def tensor_stats(x: torch.Tensor):
            x_float = x.detach().float()
            finite = torch.isfinite(x_float)
            return {
                "shape": tuple(x.shape),
                "dtype": str(x.dtype),
                "finite": int(finite.sum().item()),
                "numel": x.numel(),
                "abs_mean": round(float(x_float.abs().mean().item()), 8),
                "abs_max": round(float(x_float.abs().max().item()), 8),
                "rms": round(
                    float(torch.sqrt(x_float.square().mean()).item()), 8
                ),
            }

        if do_probe:
            valid = swa_loc >= 0
            feature_width = main_hidden.shape[-1] // self.num_target_features
            hidden_blocks = [
                tensor_stats(
                    main_hidden[
                        valid,
                        feature_id * feature_width : (feature_id + 1)
                        * feature_width,
                    ]
                )
                for feature_id in range(self.num_target_features)
            ]
            logger.warning(
                "DSpark KV pipeline: rid=%s phase=%s positions=%s swa_loc=%s "
                "target_hidden=%s valid_hidden=%s invalid_hidden=%s "
                "hidden_blocks_valid=%s main_x=%s valid_main_x=%s",
                probe_key,
                probe_phase,
                positions.detach().cpu().tolist(),
                swa_loc.detach().cpu().tolist(),
                tensor_stats(main_hidden),
                tensor_stats(main_hidden[valid]),
                (
                    tensor_stats(main_hidden[~valid])
                    if bool((~valid).any().item())
                    else None
                ),
                hidden_blocks,
                tensor_stats(main_x),
                tensor_stats(main_x[valid]),
            )

        for stage, kv in zip(self.stages, kvs):
            attn = stage.self_attn
            expected = None
            if do_probe:
                # Independent reference for RMSNorm + interleaved RoPE. This
                # intentionally does not call torch_npu.npu_rotary_mul.
                normalized = kv.detach().float()
                normalized = normalized * torch.rsqrt(
                    normalized.square().mean(dim=-1, keepdim=True) + attn.eps
                )
                normalized = (
                    normalized * attn.kv_norm.weight.detach().float()
                ).to(kv.dtype)

                rope_dim = int(attn.freqs_cis.shape[-1]) * 2
                expected = normalized.clone()
                rope = normalized[..., -rope_dim:].float().unflatten(-1, (-1, 2))
                freqs_real = attn.freqs_cis.real.contiguous()[positions].float()
                freqs_imag = attn.freqs_cis.imag.contiguous()[positions].float()
                real = rope[..., 0]
                imag = rope[..., 1]
                rotated = torch.stack(
                    (
                        real * freqs_real - imag * freqs_imag,
                        real * freqs_imag + imag * freqs_real,
                    ),
                    dim=-1,
                ).flatten(-2)
                expected[..., -rope_dim:] = rotated.to(expected.dtype)

            pool.set_swa_key_buffer_radix_fused_norm_rope(
                layer_id=attn.layer_id,
                swa_loc=swa_loc,
                kv=kv,
                kv_weight=attn.kv_norm.weight.data,
                eps=attn.eps,
                freqs_cis=attn.freqs_cis,
                positions=positions,
            )
            if do_probe:
                valid = swa_loc >= 0
                stored = None
                comparison = None
                if bool(valid.any().item()) and hasattr(pool, "get_swa_buffer"):
                    stored = pool.get_swa_buffer(
                        attn.layer_id, swa_loc[valid].to(torch.int64)
                    )
                    if stored.ndim == expected[valid].ndim + 1:
                        stored = stored.squeeze(-2)
                    # Compare on CPU so the diagnostic itself does not depend
                    # on another NPU reduction/cosine kernel.
                    actual = stored.detach().float().cpu()
                    reference = expected[valid].detach().float().cpu()
                    diff = actual - reference
                    comparison = {
                        "max_abs_error": round(
                            float(diff.abs().max().item()), 8
                        ),
                        "mean_abs_error": round(
                            float(diff.abs().mean().item()), 8
                        ),
                        "cosine": round(
                            float(
                                F.cosine_similarity(
                                    actual.reshape(1, -1),
                                    reference.reshape(1, -1),
                                    dim=-1,
                                ).item()
                            ),
                            10,
                        ),
                        "allclose_5e-2": bool(
                            torch.allclose(
                                actual, reference, atol=5e-2, rtol=5e-2
                            )
                        ),
                    }
                logger.warning(
                    "DSpark KV stage: rid=%s phase=%s stage=%d layer=%d "
                    "raw_kv=%s norm_rope_ref=%s stored=%s comparison=%s",
                    probe_key,
                    probe_phase,
                    stage.stage_id,
                    attn.layer_id,
                    tensor_stats(kv),
                    tensor_stats(expected),
                    tensor_stats(stored) if stored is not None else None,
                    comparison,
                )

        if do_probe:
            counts = getattr(self, "_kv_pipeline_probe_counts", {})
            counts[probe_key] = probe_count + 1
            self._kv_pipeline_probe_counts = counts

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = x.unsqueeze(1).repeat(1, self.hc_mult, 1)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del get_embedding, pp_proxy_tensors
        if input_embeds is None:
            input_embeds = self.forward_embed(input_ids)
        x = input_embeds
        for stage in self.stages:
            x = stage(positions, x, forward_batch)

        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)

    def collapse_hc_head(self, x: torch.Tensor) -> torch.Tensor:
        last = self.stages[-1]
        return hc_head_torch(
            x,
            last.hc_head_fn,
            last.hc_head_scale,
            last.hc_head_base,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def compute_base_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x_post_hc = self.collapse_hc_head(x)
        return self._logits_from_x_post_hc(x_post_hc), x_post_hc

    def _logits_from_x_post_hc(self, x_post_hc: torch.Tensor) -> torch.Tensor:
        last = self.stages[-1]
        x = last.norm(x_post_hc)
        weight = self.lm_head.weight
        if self._use_fp32_lm_head:
            local_logits = F.linear(x.float(), weight.float())
        else:
            local_logits = torch.matmul(x.to(weight.dtype), weight.T)
        if self._opt_markov_w2_tp_shard:
            return local_logits
        return gather_and_crop_vocab(local_logits, self.lm_head)

    def compute_confidence(
        self,
        *,
        anchor_tokens: torch.Tensor,
        sampled_tokens: torch.Tensor,
        x_post_hc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        confidence_head = self.confidence_head
        if confidence_head is None:
            return None
        bs = int(anchor_tokens.shape[0])
        x_post_hc = x_post_hc.view(bs, self.gamma, -1)
        if confidence_head.with_markov:
            prev_seq = torch.cat(
                [anchor_tokens.view(-1, 1), sampled_tokens[:, : self.gamma - 1]], dim=1
            )
            markov_embed_stack = self.markov_head.get_prev_embeddings(prev_seq)
        else:
            markov_embed_stack = None
        confidence_raw = confidence_head(x_post_hc, markov_embed_stack)
        confidence = confidence_head.apply_sts(confidence_raw)
        expect(_CONFIDENCE, confidence)
        return confidence

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        audit_enabled = envs.SGLANG_DSPARK_QUANT_AUDIT.get()
        audit_strict = envs.SGLANG_DSPARK_QUANT_AUDIT_STRICT.get()
        audit_counts = Counter()
        loaded_source_to_param = {}
        unexpected_weights = []
        expert_source_signatures = defaultdict(set)

        weights = list(weights)
        if any(name.endswith(".wo_a.scale") for name, _ in weights):
            weights = list(_dequant_fp8_wo_a(weights))

        if audit_enabled:
            self._audit_quant_methods()

        stacked_params_mapping = DEEPSEEK_V4_STACKED_PARAMS_MAPPING
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        for name, loaded_weight in weights:
            audit_counts["checkpoint_total"] += 1
            self._record_expert_source_signature(
                name=name, signatures=expert_source_signatures
            )
            mapped = self._remap_dspark_weight_name(name)
            if mapped is None:
                audit_counts["ignored"] += 1
                continue

            audit_counts["draft_selected"] += 1
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped:
                    continue
                candidate = mapped.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(candidate)
                loaded_source_to_param[name] = candidate
                audit_counts["loaded_stacked"] += 1
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in mapped:
                        continue
                    candidate = mapped.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    param = params_dict[candidate]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        candidate,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(candidate)
                    loaded_source_to_param[name] = candidate
                    audit_counts["loaded_expert"] += 1
                    break
                else:
                    if mapped not in params_dict:
                        audit_counts["unexpected"] += 1
                        unexpected_weights.append((name, mapped))
                        logger.warning(
                            "DSpark V4 draft: unexpected weight %r -> %r", name, mapped
                        )
                        continue
                    param = params_dict[mapped]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(mapped)
                    loaded_source_to_param[name] = mapped
                    audit_counts["loaded_direct"] += 1

        self._assert_confidence_head_loaded(
            params_dict=params_dict, loaded_params=loaded_params
        )
        self._assert_vocab_modules_loaded(loaded_params=loaded_params)
        if get_parallel().tp_rank == 0:
            logger.info(
                "DeepSeek-V4 DSpark loaded checkpoint-local vocabulary weights: "
                "mtp.0.embed.weight -> embed_tokens.weight, "
                "mtp.%d.head.weight -> lm_head.weight",
                self.num_stages - 1,
            )
        if audit_enabled:
            self._finish_quant_audit(
                params_dict=params_dict,
                loaded_params=loaded_params,
                loaded_source_to_param=loaded_source_to_param,
                unexpected_weights=unexpected_weights,
                expert_source_signatures=expert_source_signatures,
                audit_counts=audit_counts,
                strict=audit_strict,
            )

    @staticmethod
    def _record_expert_source_signature(*, name: str, signatures) -> None:
        """Record the tensor families present for each checkpoint expert.

        Comparing these signatures catches partially exported experts even
        when all tensors that are present happen to load successfully.
        """
        parts = name.split(".")
        if (
            len(parts) < 7
            or parts[0] != "mtp"
            or not parts[1].isdigit()
            or parts[2] != "ffn"
            or parts[3] != "experts"
            or not parts[4].isdigit()
            or parts[5] not in ("w1", "w2", "w3")
        ):
            return
        signatures[(int(parts[1]), int(parts[4]))].add(
            (parts[5], ".".join(parts[6:]))
        )

    def _audit_quant_methods(self) -> None:
        rank = get_parallel().tp_rank
        found = 0
        for name, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            found += 1
            details = []
            for attr in ("scheme", "w13_scheme", "w2_scheme"):
                value = getattr(module, attr, None)
                if value is not None:
                    details.append(f"{attr}={type(value).__name__}")
            logger.warning(
                "DSpark W4A8 audit method: rank=%d module=%s module_type=%s "
                "quant_method=%s %s",
                rank,
                name,
                type(module).__name__,
                type(quant_method).__name__,
                " ".join(details),
            )
        logger.warning(
            "DSpark W4A8 audit methods summary: rank=%d quantized_modules=%d",
            rank,
            found,
        )

    @staticmethod
    def _audit_sample_tensor(tensor: torch.Tensor, max_samples: int = 4096):
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "samples": 0,
            }
        stride = max(1, flat.numel() // max_samples)
        sample = flat[::stride][:max_samples].float()
        finite = torch.isfinite(sample)
        finite_sample = sample[finite]
        if finite_sample.numel() == 0:
            return {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "samples": int(sample.numel()),
                "finite": 0,
            }
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "samples": int(sample.numel()),
            "finite": int(finite.sum().item()),
            "min": round(float(finite_sample.min().item()), 8),
            "max": round(float(finite_sample.max().item()), 8),
            "mean": round(float(finite_sample.mean().item()), 8),
            "rms": round(
                float(torch.sqrt(finite_sample.square().mean()).item()), 8
            ),
            "zero_fraction": round(
                float((finite_sample == 0).float().mean().item()), 8
            ),
            # A cheap fingerprint useful for comparing TP/EP ranks and
            # detecting identical/uninitialized expert shards.
            "fingerprint": round(float(finite_sample[:256].sum().item()), 8),
        }

    @staticmethod
    def _is_quant_audit_parameter(name: str) -> bool:
        if name.startswith(("markov_head.", "confidence_head.")):
            return True
        if not name.startswith("stages."):
            return False
        return any(
            marker in name
            for marker in (
                "main_proj",
                "self_attn.w",
                "mlp.experts.w13_",
                "mlp.experts.w2_",
            )
        )

    def _finish_quant_audit(
        self,
        *,
        params_dict,
        loaded_params,
        loaded_source_to_param,
        unexpected_weights,
        expert_source_signatures,
        audit_counts,
        strict: bool,
    ) -> None:
        rank = get_parallel().tp_rank
        missing_runtime = sorted(set(params_dict) - loaded_params)
        audit_counts["loaded_sources"] = len(loaded_source_to_param)
        audit_counts["loaded_runtime_params"] = len(loaded_params)
        audit_counts["missing_runtime_params"] = len(missing_runtime)

        signature_counts = Counter(
            frozenset(signature)
            for signature in expert_source_signatures.values()
        )
        modal_signature = (
            signature_counts.most_common(1)[0][0] if signature_counts else frozenset()
        )
        inconsistent_experts = [
            (
                key,
                sorted(modal_signature - signature),
                sorted(signature - modal_signature),
            )
            for key, signature in expert_source_signatures.items()
            if signature != modal_signature
        ]

        logger.warning(
            "DSpark W4A8 audit load summary: rank=%d counts=%s "
            "expert_groups=%d expert_signature_variants=%d "
            "inconsistent_experts=%d",
            rank,
            dict(audit_counts),
            len(expert_source_signatures),
            len(signature_counts),
            len(inconsistent_experts),
        )
        if missing_runtime:
            logger.warning(
                "DSpark W4A8 audit missing runtime params: rank=%d count=%d "
                "sample=%s",
                rank,
                len(missing_runtime),
                missing_runtime[:50],
            )
        if unexpected_weights:
            logger.warning(
                "DSpark W4A8 audit unexpected checkpoint tensors: rank=%d "
                "count=%d sample=%s",
                rank,
                len(unexpected_weights),
                unexpected_weights[:50],
            )
        if inconsistent_experts:
            logger.warning(
                "DSpark W4A8 audit inconsistent expert tensor families: rank=%d "
                "sample=%s",
                rank,
                inconsistent_experts[:20],
            )

        for name, param in params_dict.items():
            if not self._is_quant_audit_parameter(name):
                continue
            logger.warning(
                "DSpark W4A8 audit tensor: rank=%d loaded=%s name=%s stats=%s",
                rank,
                name in loaded_params,
                name,
                self._audit_sample_tensor(param),
            )

        if strict and (
            unexpected_weights or missing_runtime or inconsistent_experts
        ):
            raise ValueError(
                "DSpark W4A8 strict audit failed: "
                f"unexpected={len(unexpected_weights)}, "
                f"missing_runtime={len(missing_runtime)}, "
                f"inconsistent_experts={len(inconsistent_experts)}. "
                "See preceding 'DSpark W4A8 audit' logs."
            )

    def _assert_confidence_head_loaded(
        self, *, params_dict: dict, loaded_params: set
    ) -> None:
        if self.confidence_head is None:
            return
        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_params
        if missing:
            raise ValueError(
                f"DSpark V4 confidence head is enabled but the checkpoint is missing "
                f"{sorted(missing)}. Provide a checkpoint with trained confidence weights, "
                f"or disable the confidence head (enable_confidence_head=False)."
            )

    @staticmethod
    def _assert_vocab_modules_loaded(*, loaded_params: set) -> None:
        required = {"embed_tokens.weight", "lm_head.weight"}
        missing = required - loaded_params
        if missing:
            raise ValueError(
                "DeepSeek-V4 DSpark draft requires its MTP-local embedding and "
                f"LM-head weights, but these runtime parameters were not loaded: "
                f"{sorted(missing)}. Expected checkpoint tensors "
                "`mtp.0.embed.weight` and `mtp.<last_stage>.head.weight`."
            )

    def _remap_dspark_weight_name(self, name: str) -> Optional[str]:
        if name.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
            return None
        if "rotary_emb.inv_freq" in name:
            return None

        if not name.startswith("mtp."):
            return None
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        stage_id, rest = parts[1], parts[2]
        if not stage_id.isdigit():
            return None
        stage_idx = int(stage_id)
        if rest in ("embed.weight", "embed_tokens.weight"):
            return "embed_tokens.weight" if stage_idx == 0 else None
        if rest in ("head.weight", "lm_head.weight"):
            return "lm_head.weight" if stage_idx == self.num_stages - 1 else None
        if rest.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
            return None

        if rest.startswith("markov_head."):
            return f"markov_head.{rest[len('markov_head.'):]}"

        if rest.startswith("confidence_head."):
            if self.confidence_head is None:
                return None
            return f"confidence_head.{rest[len('confidence_head.'):]}"

        mapped_rest = rest
        if mapped_rest.startswith("attn."):
            mapped_rest = "self_attn." + mapped_rest.removeprefix("attn.")
        elif mapped_rest.startswith("ffn."):
            mapped_rest = "mlp." + mapped_rest.removeprefix("ffn.")
        elif mapped_rest.startswith("attn_norm."):
            mapped_rest = (
                "input_layernorm." + mapped_rest.removeprefix("attn_norm.")
            )
        elif mapped_rest.startswith("ffn_norm."):
            mapped_rest = (
                "post_attention_layernorm."
                + mapped_rest.removeprefix("ffn_norm.")
            )
        mapped_rest = mapped_rest.replace(".w1.", ".gate_proj.")
        mapped_rest = mapped_rest.replace(".w2.", ".down_proj.")
        mapped_rest = mapped_rest.replace(".w3.", ".up_proj.")
        mapped_rest = mapped_rest.replace(".gate.tid2eid", ".topk.tid2eid")
        mapped_rest = mapped_rest.replace(".gate.bias", ".gate.e_score_correction_bias")
        # Only standalone FP8 scale tensors use the ``weight_scale_inv``
        # runtime name.  W4A8 MoE also stores ``scale_bias`` tensors; a broad
        # string replacement would corrupt those names into
        # ``weight_scale_inv_bias`` and leave w13/w2_scale_bias uninitialized.
        if mapped_rest.endswith(".scale"):
            mapped_rest = (
                mapped_rest.removesuffix(".scale") + ".weight_scale_inv"
            )
        return f"stages.{stage_id}.{mapped_rest}"


EntryClass = [DeepseekV4ForCausalLMDSpark]
