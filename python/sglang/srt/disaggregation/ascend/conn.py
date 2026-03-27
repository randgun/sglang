import concurrent.futures
import logging
from typing import List, Tuple, Optional

import dataclasses
import struct
import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferKVChunk,
)
from sglang.srt.environ import envs
from sglang.srt.utils import get_local_ip_auto
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class AscendKVArgsRegisterInfo(KVArgsRegisterInfo):
    dequant_scale_data_ptrs: List[int]
    dequant_scale_item_len: List[int]
    dequant_unit_num: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
            dst_state_item_lens=(
                list(struct.unpack(f"{len(msg[10])//4}I", msg[10]))
                if len(msg) > 10 and len(msg[10]) > 0 else []
            ),
            dst_state_dim_per_tensor=(
                list(struct.unpack(f"{len(msg[11])//4}I", msg[11]))
                if len(msg) > 11 and len(msg[11]) > 0 else []
            ),
            # C8
            dequant_scale_data_ptrs=list(struct.unpack(f"{len(msg[12])//8}Q", msg[12])) if len(msg) > 12 else [],
            dequant_scale_item_len=int(msg[13].decode("ascii")) if len(msg) > 13 else 0,
            dequant_unit_num=int(msg[14].decode("ascii")) if len(msg) > 14 else 0,
        )

class AscendKVManager(MooncakeKVManager):
    kv_args_register_info_class = AscendKVArgsRegisterInfo
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        self.npu_c8 = envs.SGLANG_NPU_PD_ENABLE_C8.get()
        # LKL TODO remove hard code
        self.page_size = 128
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        self.engine.batch_register(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        # The Ascend backend optimize batch registration for small memory blocks.
        self.engine.batch_register(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        )
        if self.npu_c8 and self.disaggregation_mode == DisaggregationMode.DECODE:
            self.engine.batch_register(
                self.kv_args.dequant_scale_data_ptrs,
                self.kv_args.dequant_scale_data_lens,
            )

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        num_layers = len(self.kv_args.kv_data_ptrs)
        layers_params = [
            (
                self.kv_args.kv_data_ptrs[layer_id],
                dst_kv_ptrs[layer_id],
                self.kv_args.kv_item_lens[layer_id],
            )
            for layer_id in range(num_layers)
        ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def _handle_kvcache_transfer(
        self, 
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        target_rank_registration_info: AscendKVArgsRegisterInfo,
        chunked_dst_kv_indice: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> int:
        """Handle KV cache transfer with appropriate method based on backend, TP size, and C8 mode"""
        if self.is_mla_backend or (
            self.attn_tp_size == target_rank_registration_info.dst_attn_tp_size
        ):
            if self.npu_c8:
                return self.send_kvcache_c8(
                    mooncake_session_id,
                    prefill_kv_indices,
                    target_rank_registration_info.dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    target_rank_registration_info.dst_kv_item_len,
                    target_rank_registration_info.dequant_scale_data_ptrs,
                    target_rank_registration_info.dequant_scale_item_len,
                    target_rank_registration_info.dequant_unit_num,
                )
            else:
                return self.send_kvcache(
                    mooncake_session_id,
                    prefill_kv_indices,
                    target_rank_registration_info.dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    executor,
                )
        else:
            return self.send_kvcache_slice(
                mooncake_session_id,
                prefill_kv_indices,
                target_rank_registration_info.dst_kv_ptrs,
                chunked_dst_kv_indice,
                target_rank_registration_info.dst_tp_rank,
                target_rank_registration_info.dst_attn_tp_size,
                target_rank_registration_info.dst_kv_item_len,
                executor,
            )

    def send_kvcache_c8(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_kv_item_len: int,
        dequant_scale_ptrs: list[int],
        dequant_scale_item_len: int,
        dequant_unit_num: int,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )
        print(f"+++ {prefill_kv_blocks=}, {dst_kv_blocks=}, {prefill_kv_indices=}, {dst_kv_indices=}, {dst_kv_item_len=}", flush=True)

        num_layers = len(self.kv_args.kv_data_ptrs)
        layers_params = [
            (
                self.kv_args.kv_data_ptrs[layer_id],
                dst_kv_ptrs[layer_id],
                self.kv_args.kv_item_lens[layer_id],
                dst_kv_item_len,
                dequant_scale_ptrs[layer_id],
                dequant_scale_item_len,
            )
            for layer_id in range(num_layers)
        ]

        # Worker function for processing all layers in a batch
        transfer_blocks = []
        for (
            src_ptr,
            dst_ptr,
            item_len,
            dst_item_len,
            dequant_scale_ptr,
            dequant_scale_item_len,
        ) in layers_params:
            tmp_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len // 2
                length = item_len * len(prefill_index)
                dequant_scale_addr = (
                    dequant_scale_ptr + int(decode_index[0]) * dequant_scale_item_len * self.page_size
                )
                tmp_blocks.append((src_addr, dst_addr, length, dequant_scale_addr))
            transfer_blocks.extend(tmp_blocks)

        src_addrs, dst_addrs, lengths, dequant_scale_addrs = zip(*transfer_blocks)
        return self.engine.batch_transfer_with_quant_sync(
            mooncake_session_id,
            list(src_addrs),
            list(dst_addrs),
            list(lengths),
            list(dequant_scale_addrs),
            [],
            dequant_unit_num,
        )


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    def __init__(
        self,
        mgr: AscendKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()
        self.conclude_state = None
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.state_data_ptrs
            )
            # Pack state_item_lens and state_dim_per_tensor for mamba state slice transfer
            packed_state_item_lens = b"".join(
                struct.pack("I", item_len)
                for item_len in self.kv_mgr.kv_args.state_item_lens
            )
            state_dim_per_tensor = getattr(
                self.kv_mgr.kv_args, "state_dim_per_tensor", []
            )
            packed_state_dim_per_tensor = b"".join(
                struct.pack("I", dim) for dim in state_dim_per_tensor
            )
            # C8
            if envs.SGLANG_NPU_PD_ENABLE_C8.get():
                dequant_scale_data_ptrs = b"".join(
                    struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.dequant_scale_data_ptrs
                )
                dequant_scale_item_len = str(self.kv_mgr.kv_args.dequant_scale_item_lens[0]).encode("ascii")
                dequant_unit_num = str(self.kv_mgr.kv_args.dequant_unit_num).encode("ascii")
            # Note(shangming): No need to add pp rank here since decode pp size should be equal to prefill pp size or 1
            tp_rank = self.kv_mgr.kv_args.engine_rank
            kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]
            dst_tp_rank = str(tp_rank).encode("ascii")
            dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
            dst_kv_item_len = str(kv_item_len).encode("ascii")

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            base_args = [
                            "None".encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.session_id.encode("ascii"),
                            packed_kv_data_ptrs,
                            packed_aux_data_ptrs,
                            packed_state_data_ptrs,
                            dst_tp_rank,
                            dst_attn_tp_size,
                            dst_kv_item_len,
                            packed_state_item_lens,
                            packed_state_dim_per_tensor,
                        ]
            if envs.SGLANG_NPU_PD_ENABLE_C8.get():
                base_args.extend([
                            dequant_scale_data_ptrs,
                            dequant_scale_item_len,
                            dequant_unit_num,
                        ])
            with lock:
                sock.send_multipart(base_args)

class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
