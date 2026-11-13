"""CPU unit tests for the NPU engram host-offload backend (engram_offload.py).

memfabric_hybrid is faked with an in-process pool whose entry_gather applies
the same slot-segmented row addressing the AIV kernel resolves from the
registered layout, and the runtime context (parallel / schedule / exec /
device) is stubbed at the module seams — no NPU and no real wheel needed.

Covered: the chunk/addressing math against the layout formula, bit-exact
dequant vs the device-resident torch path, forward-buffer freeze / throwaway /
empty-batch behavior, the fail-fast surface, the new/old-wheel entity paths,
and the placement routing.
"""

import contextlib
import ctypes
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.arg_groups.engram_config import (
    EngramConfig,
    EngramPlacement,
    EngramReplication,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.modules import engram_offload
from sglang.srt.layers import engram as engram_mod
from sglang.srt.layers.engram import EngramEmbedding, EngramLayout
from sglang.test.test_utils import CustomTestCase

_HEAD_DIM = 64
_N_HASH_COLS = 4  # (max_ngram_size - 1) * n_heads
_NUM_EMBEDDINGS = 1000
_SLOT_GIB = 1 << 30

_LAYOUT = EngramLayout(
    max_ngram_size=3,
    layer_ids=(5, 9),
    num_embeddings=(_NUM_EMBEDDINGS, 701),
    primes=(((11, 13), (17, 19)), ((23, 29), (31, 37))),
    n_heads=2,
    head_dim=_HEAD_DIM,
)

_NODE_LOCAL_ENV_KEYS = (
    "SGLANG_NODE_LOCAL_WORLD",
    "SGLANG_NODE_LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "LOCAL_RANK",
)


# ---------------------------------------------------------------------------
# fake memfabric_hybrid
# ---------------------------------------------------------------------------


class _FakeScene:
    LOCAL = 0
    SHARED = 1


class _FakeOffloadConfig:
    def __init__(self):
        self.device_id = 0
        self.reserve_size = 0
        self.alloc_size = 0
        self.world_size = 1
        self.rank_id = 0
        self.scene = _FakeScene.LOCAL
        self.flags = 0
        self.store_url = ""


class _FakeOffloadEntity:
    """One memfabric acc_offload entity: a node pool of world_size contiguous
    slots, each physically backed by its own rank. entry_gather resolves the
    registered layout's row -> (slot, in-slot row) addressing exactly like the
    AIV kernel; malloc hands out this rank's slot inside the shared window."""

    Scene = _FakeScene
    OffloadConfig = _FakeOffloadConfig

    def __init__(self):
        self._config = None
        self._tables = []
        self._buf = None
        self._slot = 0
        self.uninitialized = False

    def initialize(self, config):
        assert self._config is None, "fake entity initialized twice"
        self._config = config
        total = config.alloc_size * config.world_size
        self._buf = (ctypes.c_char * total)()
        self._slot = config.alloc_size
        return 0

    def malloc(self, size, flags=0):
        assert self._buf is not None and size <= self._slot
        return ctypes.addressof(self._buf) + self._config.rank_id * self._slot

    def register_entry_table(self, entry_bytes, rows, block_offset):
        self._tables.append((entry_bytes, rows, block_offset))
        return len(self._tables) - 1

    def entry_gather(self, dst, ids, count, table_id, device):
        entry_bytes, rows, block_offset = self._tables[table_id]
        n = int(count) if not torch.is_tensor(count) else int(count.item())
        dst_np = dst.numpy()
        for j, row in enumerate(ids[:n].to(torch.int64).cpu().tolist()):
            slot_i, in_slot = divmod(row, rows)
            start = slot_i * self._slot + block_offset + in_slot * entry_bytes
            dst_np[j * entry_bytes : (j + 1) * entry_bytes] = np.frombuffer(
                self._buf, dtype=np.uint8, count=entry_bytes, offset=start
            )
        return 0

    def get_last_err_msg(self):
        return "fake memfabric error detail"

    def uninitialize(self):
        self.uninitialized = True


def _fake_memfabric(multi_instance: bool) -> types.ModuleType:
    default = _FakeOffloadEntity()
    if multi_instance:
        default.Offload = _FakeOffloadEntity
    module = types.ModuleType("memfabric_hybrid")
    module.offload = default
    return module


# ---------------------------------------------------------------------------
# runtime-context stubs and env control
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _without_node_local_env():
    """Hide any launcher-provided node-local index so the coordinate fallback
    branches are exercised deterministically."""
    saved = {k: os.environ.pop(k) for k in _NODE_LOCAL_ENV_KEYS if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def _default_parallel(**overrides):
    fields = dict(tp_size=4, tp_rank=0, pp_size=1, nnodes=1)
    fields.update(overrides)
    return SimpleNamespace(**fields)


_WORLD_ENV = {"SGLANG_NODE_LOCAL_WORLD": "4", "SGLANG_NODE_LOCAL_RANK": "1"}


class _OffloadHarness:
    """Inject the fake memfabric module, stub the runtime-context seams
    (parallel / schedule / exec / device), reset the process singleton, and
    control the node-local env keys. Use as a context manager."""

    def __init__(
        self,
        multi_instance: bool = True,
        parallel=None,
        chunked_prefill_size: int = 512,
        decode_max_bs: int = 64,
        node_local_env: dict | None = None,
    ):
        self.module = _fake_memfabric(multi_instance)
        self.parallel = parallel or _default_parallel()
        self.schedule = SimpleNamespace(chunked_prefill_size=chunked_prefill_size)
        self.exec_ctx = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=decode_max_bs)
                )
            )
        )
        self.node_local_env = node_local_env

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        stack = self._stack
        stack.enter_context(patch.dict(sys.modules, {"memfabric_hybrid": self.module}))
        stack.enter_context(
            patch.object(engram_offload, "get_parallel", return_value=self.parallel)
        )
        stack.enter_context(
            patch.object(engram_offload, "get_schedule", return_value=self.schedule)
        )
        stack.enter_context(
            patch.object(engram_offload, "get_exec", return_value=self.exec_ctx)
        )
        stack.enter_context(
            patch.object(engram_offload, "_current_device_id", return_value=0)
        )
        stack.enter_context(patch.object(engram_offload, "_MANAGER", None))
        if self.node_local_env is None:
            stack.enter_context(_without_node_local_env())
        else:
            stack.enter_context(patch.dict(os.environ, self.node_local_env))
        return self

    def __exit__(self, *exc_info):
        self._stack.close()
        return False


def _open_manager(layout=_LAYOUT):
    return engram_offload.get_engram_offload_manager(layout)


# ---------------------------------------------------------------------------
# pool-content helpers
# ---------------------------------------------------------------------------


def _write_all_chunks(entity, manager, table, src_u8: np.ndarray) -> None:
    """Simulate every node-local rank having flushed its chunk (the fake pool
    is one process buffer, so the test writes the peer slots directly)."""
    pool = np.frombuffer(entity._buf, dtype=np.uint8)
    for chunk_rank in range(manager.local_world):
        lo = chunk_rank * table.rows
        hi = min(lo + table.rows, table.num_embeddings)
        if lo >= hi:
            break
        start = chunk_rank * manager.slot + table.block_offset
        n = (hi - lo) * table.entry_bytes
        pool[start : start + n] = src_u8[
            lo * table.entry_bytes : hi * table.entry_bytes
        ].reshape(-1)


def _pool_rows(entity, manager, table, ids: torch.Tensor) -> torch.Tensor:
    """The pool bytes the fake kernel returns for `ids`, read back through the
    same slot-segmented formula the registered layout encodes."""
    pool = np.frombuffer(entity._buf, dtype=np.uint8)
    out = np.empty((ids.numel(), table.entry_bytes), dtype=np.uint8)
    for j, row in enumerate(ids.reshape(-1).tolist()):
        slot_i, in_slot = divmod(row, table.rows)
        start = slot_i * manager.slot + table.block_offset + in_slot * table.entry_bytes
        out[j] = pool[start : start + table.entry_bytes]
    return torch.from_numpy(out)


def _dequant_ref(weight_u8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """The device-resident path's dequant math (the `_owned_rows` byte-view
    branch), spelled out on raw bytes: the spec the offload gather must match
    bit-for-bit."""
    w = weight_u8.view(torch.float8_e4m3fn).float()
    s = engram_mod._e8m0_to_float(scale_u8)
    return (
        (w.unflatten(-1, (-1, FP8_BLOCK_SIZE)) * s.unsqueeze(-1))
        .flatten(-2)
        .to(torch.bfloat16)
    )


def _make_backend(holder: nn.Module, layer_id: int = 5):
    backend = engram_offload.NpuEngramOffloadBackend(_LAYOUT, _HEAD_DIM, layer_id)
    backend.install_params(holder)
    return backend


# ---------------------------------------------------------------------------
# pool open + addressing
# ---------------------------------------------------------------------------


class TestPoolOpenAndAddressing(CustomTestCase):
    def test_chunk_math_and_full_range_addressing(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV) as h:
            manager = _open_manager()
            self.assertEqual((manager.local_world, manager.rank), (4, 1))
            t0, t1 = manager.table(0), manager.table(1)
            # ceil sharding: 1000/4 and ceil(701/4)
            self.assertEqual((t0.rows, t0.block_offset), (250, 0))
            self.assertEqual((t1.rows, t1.block_offset), (176, 250 * _HEAD_DIM))
            self.assertEqual(manager.slot, _SLOT_GIB)
            self.assertEqual(t0.entry_bytes, _HEAD_DIM)

            # every row of both tables, gathered back through the fake kernel,
            # matches the source bytes: covers chunk boundaries (249/250,
            # 499/500, ...) and proves the slot tail gap is never read
            for ti, e in enumerate(_LAYOUT.num_embeddings):
                table = manager.table(ti)
                src = np.fromiter(
                    (
                        (ti * 7 + r + j) % 251
                        for r in range(e)
                        for j in range(_HEAD_DIM)
                    ),
                    dtype=np.uint8,
                ).reshape(e, _HEAD_DIM)
                _write_all_chunks(manager._entity, manager, table, src)
                ids = torch.arange(e, dtype=torch.int64)
                staging = torch.empty(e * _HEAD_DIM, dtype=torch.uint8)
                self.assertEqual(
                    manager.entry_gather(
                        staging,
                        ids,
                        torch.tensor(e, dtype=torch.int32),
                        ti,
                        torch.device("cpu"),
                    ),
                    0,
                )
                np.testing.assert_array_equal(
                    staging.numpy().reshape(e, _HEAD_DIM), src
                )

    def test_dedicated_entity_on_multi_instance_wheel(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV) as h:
            manager = _open_manager()
            # the engram pool gets its own entity, not the module-level
            # default the HiCache L2 host pool would initialize
            self.assertIsNot(manager._entity, h.module.offload)

    def test_default_entity_fallback_on_old_wheel(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            manager = _open_manager()
            self.assertIs(manager._entity, h.module.offload)
            self.assertFalse(h.module.offload.uninitialized)
            manager.uninitialize()
            self.assertTrue(h.module.offload.uninitialized)

    def test_store_url_reaches_the_pool_config(self):
        with (
            _OffloadHarness(node_local_env=_WORLD_ENV),
            envs.SGLANG_ENGRAM_OFFLOAD_STORE_URL.override("tcp://127.0.0.1:8500"),
        ):
            _open_manager()
            config = engram_offload._MANAGER._entity._config
            self.assertEqual(config.store_url, "tcp://127.0.0.1:8500")
            self.assertEqual(config.scene, _FakeScene.SHARED)
            self.assertEqual(config.flags, 0)
            self.assertEqual(config.world_size, 4)
            self.assertEqual(config.rank_id, 1)
            self.assertEqual(config.alloc_size, config.reserve_size)


# ---------------------------------------------------------------------------
# pool coordinates
# ---------------------------------------------------------------------------


class TestPoolCoords(CustomTestCase):
    def test_coord_fallback_order(self):
        cases = [
            # (env, expected): SGLANG_ node-local wins over torchrun's, and a
            # single-node run falls back to the TP group when neither exists
            (
                {
                    "SGLANG_NODE_LOCAL_WORLD": "8",
                    "SGLANG_NODE_LOCAL_RANK": "3",
                    "LOCAL_WORLD_SIZE": "4",
                    "LOCAL_RANK": "1",
                },
                (8, 3),
            ),
            ({"LOCAL_WORLD_SIZE": "4", "LOCAL_RANK": "2"}, (4, 2)),
            (None, (4, 3)),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                with _OffloadHarness(node_local_env=env):
                    self.assertEqual(
                        engram_offload._node_local_pool_coords(4, 0), expected
                    )

    def test_multi_node_without_coords_fails_fast(self):
        with _OffloadHarness(parallel=_default_parallel(nnodes=2)):
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            self.assertIn("node-local", str(cm.exception))

    def test_same_node_pp_fails_fast(self):
        with _OffloadHarness(parallel=_default_parallel(pp_size=2)):
            with self.assertRaises(AssertionError):
                _open_manager()


# ---------------------------------------------------------------------------
# fail-fast surface
# ---------------------------------------------------------------------------


class TestFailFastSurface(CustomTestCase):
    def test_missing_memfabric_fails_fast_with_remedy(self):
        with _OffloadHarness(), patch.dict(sys.modules, {"memfabric_hybrid": None}):
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            message = str(cm.exception)
            self.assertIn("MEMFABRIC_HYBRID_EXTEND_LIB_PATH", message)
            self.assertIn("--engram-config", message)

    def test_old_wheel_conflicts_with_hicache_memfabric(self):
        with (
            _OffloadHarness(multi_instance=False),
            patch.object(
                engram_offload, "memfabric_host_memory_enabled", return_value=True
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            self.assertIn("SGLANG_HICACHE_HOST_MEM_BACKEND", str(cm.exception))

    def test_old_wheel_without_hicache_warns_and_uses_default(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            with self.assertLogs(
                "sglang.srt.hardware_backend.npu.modules.engram_offload",
                level="WARNING",
            ):
                _open_manager()
            self.assertIs(engram_offload._MANAGER._entity, h.module.offload)

    def test_initialize_failure_carries_error_detail(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            h.module.offload.initialize = lambda config: -3
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            message = str(cm.exception)
            self.assertIn("-3", message)
            self.assertIn("fake memfabric error detail", message)

    def test_malloc_failure_fails_fast(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            h.module.offload.malloc = lambda size, flags=0: 0
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            self.assertIn("host DRAM", str(cm.exception))

    def test_table_id_drift_fails_fast(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            original = h.module.offload.register_entry_table

            def drifting_register(entry_bytes, rows, block_offset):
                return original(entry_bytes, rows, block_offset) + 100

            h.module.offload.register_entry_table = drifting_register
            with self.assertRaises(AssertionError) as cm:
                _open_manager()
            self.assertIn("diverged", str(cm.exception))

    def test_oversized_row_fails_fast(self):
        layout = EngramLayout(
            max_ngram_size=2,
            layer_ids=(0,),
            num_embeddings=(10,),
            primes=(((13, 17),),),
            n_heads=1,
            head_dim=engram_offload._MAX_ENTRY_BYTES + 1,
        )
        with _OffloadHarness(
            node_local_env={
                "SGLANG_NODE_LOCAL_WORLD": "1",
                "SGLANG_NODE_LOCAL_RANK": "0",
            }
        ):
            with self.assertRaises(AssertionError) as cm:
                _open_manager(layout)
            self.assertIn("entry_gather limit", str(cm.exception))


# ---------------------------------------------------------------------------
# backend lifecycle
# ---------------------------------------------------------------------------


class TestBackendLifecycle(CustomTestCase):
    def test_install_params_surface(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            self.assertTrue(backend.full_replica)
            self.assertEqual((backend.rows, backend.row_start), (250, 250))
            self.assertEqual(holder.weight.shape, (250, _HEAD_DIM))
            self.assertEqual(holder.weight.device.type, "meta")
            self.assertEqual(
                holder.scale.shape, (_NUM_EMBEDDINGS, _HEAD_DIM // FP8_BLOCK_SIZE)
            )
            self.assertIsNotNone(backend._staging_weight)
            self.assertFalse(backend._finalized)
            # the framework module mirrors the chunk geometry
            self.assertEqual((holder.rows, holder.row_start), (250, 250))

    def test_weight_loader_stages_rank_slice_and_zeroes_tail(self):
        # table 1: E=701 over 4 slots -> 176 rows/slot; rank 3 owns rows
        # [528, 701) = 173 rows, so the 3-row staging tail must be zeroed
        rank_env = {"SGLANG_NODE_LOCAL_WORLD": "4", "SGLANG_NODE_LOCAL_RANK": "3"}
        with _OffloadHarness(node_local_env=rank_env):
            holder = nn.Module()
            backend = _make_backend(holder, layer_id=9)
            self.assertEqual((backend.rows, backend.row_start), (176, 528))
            full = np.array(
                [(r + j) % 251 for r in range(701) for j in range(_HEAD_DIM)],
                dtype=np.uint8,
            ).reshape(701, _HEAD_DIM)
            backend._load_weight_rows(
                holder.weight, torch.from_numpy(full).view(torch.float8_e4m3fn)
            )
            staged = backend._staging_weight.view(torch.uint8).numpy()
            np.testing.assert_array_equal(staged[:173], full[528:701])
            self.assertEqual(int(staged[173:].sum()), 0)

    def test_finish_flushes_pool_chunk_and_freezes_buffers(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV) as h:
            holder = nn.Module()
            backend = _make_backend(holder)
            full = torch.randint(
                0, 256, (_NUM_EMBEDDINGS, _HEAD_DIM), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            holder.weight.weight_loader(holder.weight, full)
            backend.finish_load("layer 5")
            self.assertTrue(backend._finalized)
            self.assertIsNone(backend._staging_weight)
            # buffers frozen for the largest configured batch
            self.assertEqual(backend._buffers.capacity, 512 * _N_HASH_COLS)
            # this rank's pool chunk holds exactly its checkpoint slice
            entity = backend._manager._entity
            pool = np.frombuffer(entity._buf, dtype=np.uint8)
            start = backend._manager.rank * backend._manager.slot
            start += backend._table.block_offset
            n = backend.rows * backend._table.entry_bytes
            expected = (
                full.view(torch.uint8)
                .numpy()[
                    backend.row_start * _HEAD_DIM : (backend.row_start + backend.rows)
                    * _HEAD_DIM
                ]
                .reshape(-1)
            )
            np.testing.assert_array_equal(pool[start : start + n], expected)
            # a second finish_load is a no-op
            backend.finish_load("layer 5")
            self.assertEqual(backend._buffers.capacity, 512 * _N_HASH_COLS)

    def test_gather_before_finish_raises(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            with self.assertRaises(AssertionError):
                backend.gather(torch.zeros(2, _N_HASH_COLS, dtype=torch.int64))


# ---------------------------------------------------------------------------
# gather semantics
# ---------------------------------------------------------------------------


class TestGatherEquivalence(CustomTestCase):
    def test_gather_matches_device_path_bitwise(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            gen = torch.Generator().manual_seed(7)
            # e4m3fn has exactly two NaN encodings (0x7F / 0xFF); keep them out
            # so torch.equal's element-wise == stays meaningful
            full_u8 = torch.randint(
                0, 256, (_NUM_EMBEDDINGS, _HEAD_DIM), dtype=torch.uint8, generator=gen
            )
            full_u8[full_u8 == 0x7F] = 0
            full_u8[full_u8 == 0xFF] = 0
            full = full_u8.view(torch.float8_e4m3fn)
            full_scale = torch.randint(
                0,
                256,
                (_NUM_EMBEDDINGS, _HEAD_DIM // FP8_BLOCK_SIZE),
                dtype=torch.uint8,
                generator=gen,
            ).view(torch.float8_e8m0fnu)
            holder.weight.weight_loader(holder.weight, full)
            holder.scale.weight_loader(holder.scale, full_scale)
            backend.finish_load("layer 5")
            # fill every rank's chunk so the gather can serve arbitrary ids
            _write_all_chunks(
                backend._manager._entity,
                backend._manager,
                backend._table,
                full.view(torch.uint8).numpy(),
            )

            ids = torch.randint(0, _NUM_EMBEDDINGS, (17, _N_HASH_COLS), generator=gen)
            out = backend.gather(ids)
            self.assertEqual(out.shape, (17, _N_HASH_COLS, _HEAD_DIM))
            self.assertEqual(out.dtype, torch.bfloat16)

            # spec: dequant of exactly the bytes the pool addressing returns
            staged = _pool_rows(
                backend._manager._entity, backend._manager, backend._table, ids
            )
            ref = _dequant_ref(staged, full_scale.view(torch.uint8)[ids])
            self.assertTrue(torch.equal(out, ref))

            # same bytes through the device-resident path's torch branch
            with (
                patch.object(
                    engram_mod, "resolve_engram_config", return_value=EngramConfig()
                ),
                patch.object(engram_mod, "_needs_fp8_byte_view", return_value=True),
                patch.object(
                    engram_mod,
                    "get_parallel",
                    return_value=SimpleNamespace(tp_size=1, tp_rank=0),
                ),
            ):
                resident = EngramEmbedding(
                    _NUM_EMBEDDINGS, _HEAD_DIM, 5, column_bounds=(0, _NUM_EMBEDDINGS)
                )
                resident.weight.data.copy_(full)
                resident.scale.data.copy_(full_scale)
                device_path = resident._owned_rows(ids)
            self.assertTrue(torch.equal(out, device_path))

    def test_capture_oversize_fails_fast(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            backend.finish_load("layer 5")
            self.assertEqual(backend._buffers.capacity, 512 * _N_HASH_COLS)
            ids = torch.zeros((512 * _N_HASH_COLS + 1, 1), dtype=torch.int64)
            with (
                patch.object(
                    engram_offload, "is_device_stream_capturing", return_value=True
                ),
                self.assertRaises(RuntimeError) as cm,
            ):
                backend.gather(ids)
            self.assertIn("undersized", str(cm.exception))

    def test_eager_oversize_uses_throwaway_and_keeps_frozen(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            backend.finish_load("layer 5")
            ids = torch.zeros(600 * _N_HASH_COLS, dtype=torch.int64).reshape(
                600, _N_HASH_COLS
            )
            out = backend.gather(ids)
            self.assertEqual(out.shape, (600, _N_HASH_COLS, _HEAD_DIM))
            # the frozen buffer is intact for graph replays
            self.assertEqual(backend._buffers.capacity, 512 * _N_HASH_COLS)

    def test_empty_batch_skips_gather_and_sizing(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            backend.finish_load("layer 5")
            calls = backend._calls
            out = backend.gather(torch.empty(0, _N_HASH_COLS, dtype=torch.int64))
            self.assertEqual(out.shape, (0, _N_HASH_COLS, _HEAD_DIM))
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(backend._calls, calls)
            self.assertEqual(backend._buffers.capacity, 512 * _N_HASH_COLS)


# ---------------------------------------------------------------------------
# placement routing (engram.py side, no pool involved)
# ---------------------------------------------------------------------------


def _patch_engram_config(config: EngramConfig):
    """Pin the engram config the layers read. The scheduler-side hook resolves
    --engram-config (or the legacy envs) into it before model construction."""
    return patch.object(engram_mod, "resolve_engram_config", return_value=config)


@contextlib.contextmanager
def _routing_patches(config, is_arch35: bool):
    """NPU seams for EngramEmbedding routing: the pinned engram config, the
    platform, the arch35 probe, and a two-rank TP group."""
    with contextlib.ExitStack() as stack:
        for cm in (
            _patch_engram_config(config),
            patch.object(engram_mod, "is_npu", return_value=True),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=is_arch35,
            ),
            patch.object(
                engram_mod,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=2, tp_rank=1),
            ),
        ):
            stack.enter_context(cm)
        yield


class TestPlacementRouting(CustomTestCase):
    def test_gpu_placement_keeps_device_tables(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with _routing_patches(EngramConfig(), is_arch35=True):
                embed = EngramEmbedding(
                    _NUM_EMBEDDINGS,
                    _HEAD_DIM,
                    5,
                    column_bounds=(0, _NUM_EMBEDDINGS),
                )
                self.assertIsNone(embed._offload_backend)
                self.assertIsNone(embed.host_table)

    def test_cpu_placement_engages_offload_backend(self):
        config = EngramConfig(
            placement=EngramPlacement.CPU, replication=EngramReplication.PER_NODE
        )
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with _routing_patches(config, is_arch35=True):
                embed = EngramEmbedding(
                    _NUM_EMBEDDINGS,
                    _HEAD_DIM,
                    5,
                    column_bounds=(0, _NUM_EMBEDDINGS),
                    layout=_LAYOUT,
                )
                self.assertIsNotNone(embed._offload_backend)
                self.assertIsNone(embed.host_table)
                self.assertEqual(embed.weight.device.type, "meta")
                self.assertEqual((embed.rows, embed.row_start), (250, 250))

    def test_pre_arch35_degrades_to_device_tables(self):
        config = EngramConfig(placement=EngramPlacement.CPU)
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with _routing_patches(config, is_arch35=False):
                with self.assertLogs("sglang.srt.layers.engram", level="WARNING"):
                    embed = EngramEmbedding(
                        _NUM_EMBEDDINGS,
                        _HEAD_DIM,
                        5,
                        column_bounds=(0, _NUM_EMBEDDINGS),
                    )
                self.assertIsNone(embed._offload_backend)
                self.assertIsNone(embed.host_table)

    def test_replication_none_warns_node_shared(self):
        # The acc_offload pool is always node-shared; replication 'none' only
        # draws a warning, it must not change the backend.
        config = EngramConfig(placement=EngramPlacement.CPU)
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with _routing_patches(config, is_arch35=True):
                with self.assertLogs("sglang.srt.layers.engram", level="WARNING"):
                    embed = EngramEmbedding(
                        _NUM_EMBEDDINGS,
                        _HEAD_DIM,
                        5,
                        column_bounds=(0, _NUM_EMBEDDINGS),
                        layout=_LAYOUT,
                    )
                self.assertIsNotNone(embed._offload_backend)

    def test_missing_layout_fails_fast(self):
        config = EngramConfig(
            placement=EngramPlacement.CPU, replication=EngramReplication.PER_NODE
        )
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with _routing_patches(config, is_arch35=True):
                with self.assertRaises(AssertionError) as cm:
                    EngramEmbedding(
                        _NUM_EMBEDDINGS,
                        _HEAD_DIM,
                        5,
                        column_bounds=(0, _NUM_EMBEDDINGS),
                    )
                self.assertIn("EngramLayout", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
