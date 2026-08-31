import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="base-a-test-1-npu-a2")


def _step_reference(loc, seq_lens, *, topk, num_steps, step_id, ratio):
    if loc is None or loc.numel() == 0:
        return loc
    positions = seq_lens[:, None, None].to(torch.int64) + torch.arange(num_steps)
    positions = positions.expand(-1, topk, -1)
    selected = ((positions + 1) % ratio) == 0
    counts = selected.reshape(-1).to(torch.int64)
    offsets = torch.cumsum(counts, dim=0) - counts
    step_mask = selected[:, :, step_id].reshape(-1)
    step_offsets = offsets.reshape(seq_lens.numel(), topk, num_steps)[
        :, :, step_id
    ].reshape(-1)
    return loc[step_offsets[step_mask].to(torch.int64)]


class TestDsv4MetadataTriton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("requires an Ascend NPU")

    def test_step_compress_matches_stable_torch_reference(self):
        from sglang.srt.hardware_backend.npu.attention.dsv4_metadata import (
            step_compress,
        )

        seq_lens_cpu = torch.tensor([3, 7, 126, 127], dtype=torch.int32)
        topk, num_steps = 2, 3
        lanes = seq_lens_cpu.numel() * topk
        full_cpu = torch.arange(lanes * num_steps, dtype=torch.int32)
        swa_cpu = torch.arange(100, 100 + lanes * num_steps, dtype=torch.int64)
        c4_count = topk * sum(
            (int(seq_len) + num_steps) // 4 - int(seq_len) // 4
            for seq_len in seq_lens_cpu
        )
        c128_count = topk * sum(
            (int(seq_len) + num_steps) // 128 - int(seq_len) // 128
            for seq_len in seq_lens_cpu
        )
        c4_cpu = torch.arange(1000, 1000 + c4_count, dtype=torch.int64)
        c128_cpu = torch.arange(2000, 2000 + c128_count, dtype=torch.int64)

        for step_id in range(num_steps - 1):
            full, swa, c4, c128 = step_compress(
                full_cpu.npu(),
                swa_cpu.npu(),
                c4_cpu.npu(),
                c128_cpu.npu(),
                seq_lens_cpu.npu(),
                seq_lens_cpu,
                raw_bs=seq_lens_cpu.numel(),
                topk=topk,
                num_steps=num_steps,
                step_id=step_id,
            )
            expected_full = full_cpu.reshape(-1, num_steps)[:, step_id]
            expected_swa = swa_cpu.reshape(-1, num_steps)[:, step_id]
            expected_c4 = _step_reference(
                c4_cpu,
                seq_lens_cpu,
                topk=topk,
                num_steps=num_steps,
                step_id=step_id,
                ratio=4,
            )
            expected_c128 = _step_reference(
                c128_cpu,
                seq_lens_cpu,
                topk=topk,
                num_steps=num_steps,
                step_id=step_id,
                ratio=128,
            )
            self.assertTrue(torch.equal(full.cpu(), expected_full))
            self.assertTrue(torch.equal(swa.cpu(), expected_swa))
            self.assertTrue(torch.equal(c4.cpu(), expected_c4))
            self.assertTrue(torch.equal(c128.cpu(), expected_c128))

    def test_explicit_state_refresh_matches_ragged_reference(self):
        from sglang.srt.hardware_backend.npu.attention.dsv4_metadata import (
            _refresh_graph_explicit_state_block,
        )

        req_to_token_cpu = torch.arange(96, dtype=torch.int32).reshape(3, 32)
        req_to_token_cpu[1, 0] = -1
        mapping_cpu = torch.arange(96, dtype=torch.int64) * 2
        mapping_cpu = torch.cat((mapping_cpu, torch.tensor([-1], dtype=torch.int64)))
        token_pool = SimpleNamespace(full_to_swa_index_mapping=mapping_cpu.npu())
        req_pool_indices_cpu = torch.tensor([1, 2], dtype=torch.int32)
        start_pos_cpu = torch.tensor([4, 10], dtype=torch.int32)
        seqused_cpu = torch.tensor([4, 2], dtype=torch.int32)
        cu_seqlens_cpu = torch.tensor([0, 4, 6], dtype=torch.int32)

        for ratio, ring_size, swa_page_size, dummy in (
            (4, 8, 16, 255),
            (128, 128, 16, 511),
        ):
            history = (2 if ratio == 4 else 1) * ratio
            width = history + 4
            dst = torch.empty((2, width), dtype=torch.int32, device="npu")
            state_pool = SimpleNamespace(
                ring_size=ring_size,
                swa_page_size=swa_page_size,
                dummy_state_loc=dummy,
            )
            _refresh_graph_explicit_state_block(
                dst,
                compress_ratio=ratio,
                state_pool=state_pool,
                token_to_kv_pool=token_pool,
                req_to_token=req_to_token_cpu.npu(),
                req_pool_indices=req_pool_indices_cpu.npu(),
                start_pos=start_pos_cpu.npu(),
                seqused=seqused_cpu.npu(),
                cu_seqlens=cu_seqlens_cpu.npu(),
            )

            columns = torch.arange(width, dtype=torch.int64)
            positions = start_pos_cpu[:, None].to(torch.int64) - history + columns
            capacities = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
            valid = (
                (seqused_cpu[:, None] > 0)
                & (positions >= 0)
                & (columns[None, :] < history + capacities[:, None])
            )
            if ratio == 4:
                safe_positions = positions.clamp(0, req_to_token_cpu.shape[1] - 1)
                full_locs = req_to_token_cpu[
                    req_pool_indices_cpu[:, None].to(torch.int64), safe_positions
                ]
                swa_locs = mapping_cpu[full_locs]
                state_locs = (swa_locs // swa_page_size) * ring_size + (
                    swa_locs % ring_size
                )
                valid = valid & (swa_locs >= 0)
            else:
                state_locs = (
                    req_pool_indices_cpu[:, None] * ring_size + positions % ring_size
                )
            expected = torch.where(valid, state_locs.to(torch.int32), dummy)
            self.assertTrue(torch.equal(dst.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
