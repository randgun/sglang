import logging
from typing import Dict, Optional, Tuple

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.runtime_context import get_parallel
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildCommitInjectLayout,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

logger = logging.getLogger(__name__)


class TargetHiddenKvInjector:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        model_runner,
        device,
        verify_num_draft_tokens: int,
        block_pos_offsets: torch.Tensor,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.model_runner = model_runner
        self.device = device
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self._block_pos_offsets = block_pos_offsets
        # Debug-only expected fingerprints, keyed by (layer_id, swa_loc).
        self._kv_probe_expected: Dict[Tuple[int, int], tuple] = {}
        self._kv_probe_read_counts: Dict[str, int] = {}
        self._kv_probe_inject_counts: Dict[str, int] = {}

    @staticmethod
    def _is_kv_probe_request(probe_key: Optional[str]) -> bool:
        return (
            probe_key is not None
            and probe_key.startswith(("dspark-kv-debug", "dspark-token-debug"))
            and get_parallel().tp_rank == 0
        )

    def _draft_layer_ids(self):
        layer_ids = []
        for stage_id, stage in enumerate(getattr(self.draft_model, "stages", ())):
            attn = getattr(stage, "self_attn", None)
            layer_ids.append(int(getattr(attn, "layer_id", stage_id)))
        return layer_ids

    def _should_probe_inject(self, probe_key: Optional[str]) -> bool:
        return self._is_kv_probe_request(probe_key) and (
            self._kv_probe_inject_counts.get(probe_key, 0) < 3
        )

    def _snapshot_swa_slots(self, pool, swa_loc: torch.Tensor):
        """Return lightweight byte fingerprints for the last few valid SWA slots."""
        valid_locs = swa_loc[swa_loc >= 0].to(dtype=torch.int64)
        if valid_locs.numel() == 0:
            return [], {}

        # The history tail is the most useful part for checking the next draft step.
        selected_locs = valid_locs[-8:].detach().cpu().tolist()
        page_size = int(pool.swa_kv_pool.page_size)

        snapshots = {}
        for layer_id in self._draft_layer_ids():
            raw = pool.get_swa_raw_buffer(layer_id)
            layer_snapshot = {}
            if raw.ndim >= 3:
                # Ascend PA_ND layout: [num_pages, page_size, num_heads, dim].
                # SWA locations are flat token indices across the first two axes.
                loc_ids = torch.tensor(
                    selected_locs, dtype=torch.int64, device=raw.device
                )
                slots = (
                    raw.flatten(0, 1).index_select(0, loc_ids).detach().float().cpu()
                )
                for i, loc in enumerate(selected_locs):
                    slot = slots[i]
                    layer_snapshot[int(loc)] = (
                        round(float(slot.abs().sum().item()), 6),
                        int(torch.count_nonzero(slot).item()),
                        round(float(slot.abs().max().item()), 6),
                    )
            else:
                # Packed byte layouts differ across backends. Fingerprint the
                # containing page, which is sufficient to prove that a write
                # reached the expected physical page without assuming offsets.
                pages = sorted({int(loc) // page_size for loc in selected_locs})
                page_index = {page: i for i, page in enumerate(pages)}
                page_ids = torch.tensor(
                    pages, dtype=torch.int64, device=raw.device
                )
                page_rows = raw.index_select(0, page_ids).detach().cpu()
                page_fingerprints = {}
                for page, i in page_index.items():
                    row = page_rows[i]
                    page_fingerprints[page] = (
                        round(float(row.float().abs().sum().item()), 6),
                        int(torch.count_nonzero(row).item()),
                        round(float(row.float().abs().max().item()), 6),
                    )
                for loc in selected_locs:
                    layer_snapshot[int(loc)] = page_fingerprints[
                        int(loc) // page_size
                    ]
            snapshots[layer_id] = layer_snapshot
        return selected_locs, snapshots

    def _probe_inject(
        self,
        *,
        probe_key: Optional[str],
        probe_phase: str,
        cache_loc: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_rows: int,
        before,
        after,
    ) -> None:
        if not self._is_kv_probe_request(probe_key):
            return
        selected_locs, after_snapshot = after
        _, before_snapshot = before
        valid_mask = swa_loc >= 0
        full_tail = cache_loc[valid_mask][-8:].detach().cpu().tolist()
        pos_tail = positions[valid_mask][-8:].detach().cpu().tolist()
        changed = {}
        for layer_id, slots in after_snapshot.items():
            changed[layer_id] = {
                loc: before_snapshot.get(layer_id, {}).get(loc) != fingerprint
                for loc, fingerprint in slots.items()
            }
            for loc, fingerprint in slots.items():
                self._kv_probe_expected[(layer_id, loc)] = fingerprint

        logger.warning(
            "DSpark KV inject: rid=%s phase=%s full_tail=%s swa_tail=%s "
            "positions_tail=%s rows=(hidden=%d,loc=%d,pos=%d) "
            "valid=%d unique_valid=%d layout=(shape=%s,dtype=%s,page_size=%d,"
            "kv_dim=%s) changed=%s fingerprints=%s",
            probe_key,
            probe_phase,
            full_tail,
            selected_locs,
            pos_tail,
            target_hidden_rows,
            swa_loc.numel(),
            positions.numel(),
            int(valid_mask.sum().item()),
            int(torch.unique(swa_loc[swa_loc >= 0]).numel()),
            tuple(
                self.draft_model_runner.token_to_kv_pool.get_swa_raw_buffer(
                    self._draft_layer_ids()[0]
                ).shape
            ),
            self.draft_model_runner.token_to_kv_pool.get_swa_raw_buffer(
                self._draft_layer_ids()[0]
            ).dtype,
            int(self.draft_model_runner.token_to_kv_pool.swa_kv_pool.page_size),
            getattr(
                self.draft_model_runner.token_to_kv_pool.swa_kv_pool,
                "kv_cache_total_dim",
                None,
            ),
            changed,
            after_snapshot,
        )
        self._kv_probe_inject_counts[probe_key] = (
            self._kv_probe_inject_counts.get(probe_key, 0) + 1
        )

    def probe_request_history(
        self, *, batch: ScheduleBatch, probe_phase: str = "draft_read"
    ) -> None:
        """Check that the request history resolves to the slots last injected."""
        probe_key = str(batch.reqs[0].rid) if batch.reqs else ""
        if not self._is_kv_probe_request(probe_key):
            return
        count = self._kv_probe_read_counts.get(probe_key, 0)
        if count >= 3:
            return

        req_idx = int(batch.req_pool_indices[0].detach().cpu().item())
        seq_len = int(batch.seq_lens[0].detach().cpu().item())
        full_locs = self.model_runner.req_to_token_pool.req_to_token[
            req_idx, :seq_len
        ]
        pool = self.draft_model_runner.token_to_kv_pool
        swa_locs = pool.translate_loc_from_full_to_swa(full_locs).to(torch.int64)
        selected_locs, snapshot = self._snapshot_swa_slots(pool, swa_locs)

        matches_expected = {}
        for layer_id, slots in snapshot.items():
            matches_expected[layer_id] = {
                loc: (
                    self._kv_probe_expected.get((layer_id, loc)) == fingerprint
                    if (layer_id, loc) in self._kv_probe_expected
                    else None
                )
                for loc, fingerprint in slots.items()
            }

        logger.warning(
            "DSpark KV read: rid=%s step=%d phase=%s seq_len=%d "
            "full_tail=%s swa_tail=%s matches_injected=%s fingerprints=%s",
            probe_key,
            count,
            probe_phase,
            seq_len,
            full_locs[-8:].detach().cpu().tolist(),
            selected_locs,
            matches_expected,
            snapshot,
        )
        self._kv_probe_read_counts[probe_key] = count + 1

    def inject_target_hidden(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        probe_key: Optional[str] = None,
        probe_phase: str = "unknown",
    ) -> None:
        if target_hidden is None or target_hidden.numel() == 0:
            return
        device = self.model_runner.device
        cache_loc = cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)
        positions = positions.to(device=device, dtype=torch.int64, non_blocking=True)
        target_hidden = target_hidden.to(device=device, non_blocking=True)
        n_real = positions.shape[0]
        if target_hidden.shape[0] > n_real:
            target_hidden = target_hidden[:n_real]
        if cache_loc_2d is not None:
            cache_loc_2d = cache_loc_2d.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if commit_lens is not None:
            commit_lens = commit_lens.to(
                device=device, dtype=torch.int32, non_blocking=True
            )

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            self._inject_mla(
                pool=pool,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                probe_key=probe_key,
                probe_phase=probe_phase,
            )
            return

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                target_hidden=target_hidden,
                pool=pool,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )

    def _inject_mla(
        self,
        *,
        pool,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        probe_key: Optional[str],
        probe_phase: str,
    ) -> None:
        swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
        if commit_lens is not None and cache_loc_2d is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
            committed_mask = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            swa_loc = torch.where(committed_mask, swa_loc, torch.full_like(swa_loc, -1))

        do_probe = self._should_probe_inject(probe_key)
        before = self._snapshot_swa_slots(pool, swa_loc) if do_probe else None
        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
                probe_key=probe_key,
                probe_phase=probe_phase,
            )
        if do_probe:
            after = self._snapshot_swa_slots(pool, swa_loc)
            self._probe_inject(
                probe_key=probe_key,
                probe_phase=probe_phase,
                cache_loc=cache_loc,
                swa_loc=swa_loc,
                positions=positions,
                target_hidden_rows=target_hidden.shape[0],
                before=before,
                after=after,
            )

    def inject_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        hidden_strided: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
        probe_key: Optional[str] = None,
    ) -> None:
        stride = self.verify_num_draft_tokens
        prefix_lens = batch.seq_lens
        hidden = hidden_strided.view(bs, stride, -1)

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if hidden_strided.numel() == 0:
                return
            inject_layout = BuildCommitInjectLayout.execute(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                prefix_lens=prefix_lens,
                block_pos_offsets=self._block_pos_offsets[:stride],
                full_to_swa_mapping=pool.full_to_swa_index_mapping,
                commit_lens=commit_lens,
                stride=stride,
            )
            do_probe = self._should_probe_inject(probe_key)
            before = (
                self._snapshot_swa_slots(pool, inject_layout.swa_loc)
                if do_probe
                else None
            )
            with torch.inference_mode():
                self.draft_model.write_target_hidden_kv(
                    main_hidden=hidden.reshape(-1, hidden.shape[-1]),
                    swa_loc=inject_layout.swa_loc,
                    positions=inject_layout.positions,
                    pool=pool,
                    probe_key=probe_key,
                    probe_phase="verify_compact",
                )
            if do_probe:
                after = self._snapshot_swa_slots(pool, inject_layout.swa_loc)
                self._probe_inject(
                    probe_key=probe_key,
                    probe_phase="verify_compact",
                    cache_loc=inject_layout.swa_loc,
                    swa_loc=inject_layout.swa_loc,
                    positions=inject_layout.positions,
                    target_hidden_rows=hidden.reshape(-1, hidden.shape[-1]).shape[0],
                    before=before,
                    after=after,
                )
            return

        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        verify_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + stride,
            batch_size=bs,
            draft_token_num=stride,
            device=self.device,
        )
        verify_cache_loc_2d = verify_cache_loc.view(bs, stride)
        self.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_cache_loc,
            cache_loc_2d=verify_cache_loc_2d,
            positions=positions_2d.reshape(-1),
            commit_lens=commit_lens,
            probe_key=probe_key,
            probe_phase="verify_ragged_fallback",
        )
