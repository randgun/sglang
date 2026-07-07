from typing import TYPE_CHECKING, Optional

import logging

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import StandardTopKOutput, select_experts
from sglang.srt.utils.common import get_bool_env_var, get_int_env_var
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput

_DEBUG_NAN_ENV = "SGLANG_DSV4_NPU_DEBUG_NAN"
_DEBUG_MOE_ENV = "SGLANG_DSV4_NPU_DEBUG_MOE"
_DEBUG_MOE_LAYER_ENV = "SGLANG_DSV4_NPU_DEBUG_MOE_LAYER"
_DEBUG_SYNC_ENV = "SGLANG_DSV4_NPU_DEBUG_SYNC"
_DEBUG_MAX_PRINTS_ENV = "SGLANG_DSV4_NPU_DEBUG_MAX_PRINTS"
_debug_log_counts: dict[str, int] = {}

logger = logging.getLogger(__name__)


def _npu_topk_debug_enabled() -> bool:
    return get_bool_env_var(_DEBUG_MOE_ENV) or get_bool_env_var(_DEBUG_NAN_ENV)


def _npu_topk_should_probe_layer(layer_id: Optional[int]) -> bool:
    target_layer = get_int_env_var(_DEBUG_MOE_LAYER_ENV, -1)
    if target_layer < 0:
        return True
    return layer_id is not None and int(layer_id) == target_layer


def _npu_topk_full_stats(layer_id: Optional[int]) -> bool:
    return get_bool_env_var(_DEBUG_MOE_ENV) and _npu_topk_should_probe_layer(layer_id)


def _npu_topk_is_stream_capturing() -> bool:
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if get_is_capture_mode():
            return True
    except Exception:
        pass
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


def _npu_topk_log_limited(key: str, message: str) -> None:
    max_prints = get_int_env_var(_DEBUG_MAX_PRINTS_ENV, 20)
    count = _debug_log_counts.get(key, 0)
    if count >= max_prints:
        return
    _debug_log_counts[key] = count + 1
    logger.warning(message)


def _npu_topk_debug_sync(label: str) -> None:
    if not get_bool_env_var(_DEBUG_SYNC_ENV) or _npu_topk_is_stream_capturing():
        return
    torch.npu.synchronize()


def _npu_topk_probe_tensor(
    tensor: Optional[torch.Tensor],
    label: str,
    layer_id: Optional[int],
) -> None:
    if (
        tensor is None
        or not _npu_topk_debug_enabled()
        or not _npu_topk_should_probe_layer(layer_id)
        or _npu_topk_is_stream_capturing()
    ):
        return

    _npu_topk_debug_sync(label)
    probe = tensor if tensor.is_floating_point() else tensor.to(torch.float32)
    finite = torch.isfinite(probe)
    finite_count = int(finite.sum().item())
    nan_count = int(torch.isnan(probe).sum().item())
    inf_count = int(torch.isinf(probe).sum().item())
    full_stats = _npu_topk_full_stats(layer_id)
    if not full_stats and nan_count == 0 and inf_count == 0:
        return

    zero_count = int((probe == 0).sum().item())
    if finite_count > 0:
        finite_values = probe[finite].to(torch.float32)
        min_val = float(finite_values.min().item())
        max_val = float(finite_values.max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")

    invalid_rows = []
    if tensor.ndim >= 2 and (nan_count > 0 or inf_count > 0):
        flat = probe.reshape(-1, probe.shape[-1])
        invalid_mask = ~torch.isfinite(flat).all(dim=1)
        invalid_rows = invalid_mask.nonzero(as_tuple=False).flatten()[:8].cpu().tolist()

    layer = -1 if layer_id is None else int(layer_id)
    prefix = (
        "DSV4 NPU topk tensor stats: "
        if full_stats
        else "DSV4 NPU topk tensor contains invalid values: "
    )
    _npu_topk_log_limited(
        f"{'tensor' if full_stats else 'invalid'}-{layer}-{label}",
        prefix
        + f"label={label}, layer_id={layer}, "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"finite_count={finite_count}, nan_count={nan_count}, "
        f"inf_count={inf_count}, zero_count={zero_count}, "
        f"finite_min={min_val}, finite_max={max_val}, "
        f"invalid_rows={invalid_rows}",
    )


def _mask_padded_topk_rows_npu(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_token_non_padded is None:
        return topk_weights, topk_ids

    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    mask = (indices >= num_token_non_padded).unsqueeze(-1)
    topk_ids = torch.where(mask, torch.full_like(topk_ids, -1), topk_ids)
    topk_weights = torch.where(mask, torch.zeros_like(topk_weights), topk_weights)
    return topk_weights, topk_ids


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias

    _npu_topk_probe_tensor(router_logits, "topk-router-logits", layer_id)

    # Fast path: simple top-k without grouped routing and bias
    if not use_grouped_topk and correction_bias is None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=topk_config.top_k,
        )

        if renormalize:
            topk_weights = l1_norm(
                topk_weights
                if topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1]
            )
        topk_weights = topk_weights.to(torch.float32)

    # sqrtsoftplus (DSV4 noaux_tc): the NPU op only scores sigmoid/softmax, so use
    # a torch path. top-k over (scores + bias); weights from un-biased scores.
    elif topk_config.scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(router_logits.float()).sqrt()
        scores_for_choice = (
            scores + correction_bias.unsqueeze(0).float()
            if correction_bias is not None
            else scores
        )
        _, topk_ids = torch.topk(
            scores_for_choice, k=topk_config.top_k, dim=-1, sorted=False
        )
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = scores.gather(1, topk_ids)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights = topk_weights * topk_config.routed_scaling_factor
        topk_weights = topk_weights.to(torch.float32)

    # Support grouped top-k or correction bias or sigmoid or routed_scaling_factor
    elif (
        correction_bias is not None
        or topk_config.scoring_func == "sigmoid"
        or num_token_non_padded is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=topk_config.top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            # num_expert_group and topk_group in some topk_config without group is None, (not supported by this ops)
            k_group=topk_config.topk_group if use_grouped_topk else 1,
            group_count=topk_config.num_expert_group if use_grouped_topk else 1,
            group_select_mode=(1 if use_grouped_topk else 0),
            renorm=0,
            norm_type=1,  # 1 for sigmoid, 0 for softmax
            routed_scaling_factor=(
                1 if renormalize else topk_config.routed_scaling_factor
            ),
            eps=float(1e-20),
        )
        topk_weights = topk_weights.to(torch.float32)

    # torch native is not yet supported num_token_non_padded
    # Fallback to torch native implementation
    else:
        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    topk_weights, topk_ids = _mask_padded_topk_rows_npu(
        topk_weights, topk_ids, num_token_non_padded
    )
    _npu_topk_probe_tensor(topk_weights, "topk-output-weights", layer_id)
    _npu_topk_probe_tensor(topk_ids, "topk-output-ids", layer_id)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    if (cap := get_global_experts_capturer()) is not None:
        cap.capture(
            layer_id=layer_id,
            topk_indices=topk_ids,
        )

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
