import concurrent.futures
import logging
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.base.conn import KVPoll
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

logger = logging.getLogger(__name__)


class AscendKVManager(MooncakeKVManager):
    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )
        self.npu_c8 = envs.SGLANG_NPU_ENABLE_KVCACHE_C8

    def register_buffer_to_engine(self):
        self.engine.batch_register(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        # The Ascend backend optimize batch registration for small memory blocks.
        self.engine.batch_register(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        )
        if self.npu_c8:
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

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if req.mooncake_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                                break

                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        if self.is_mla_backend or (
                            self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            if self.npu_c8:
                                ret = self.send_kvcache_c8(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                )
                            else:
                                ret = self.send_kvcache(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    executor,
                                )
                        else:
                            raise NotImplementedError(
                                f"None MLA backend do not support unequal tp on NPU."
                            )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self.failed_sessions.add(req.mooncake_session_id)
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break

                        if kv_chunk.is_last:
                            if kv_chunk.state_indices is not None:
                                self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    target_rank_registration_info.dst_state_data_ptrs,
                                    executor,
                                    target_rank_registration_info,
                                )

                            # Only the last chunk we need to send the aux data
                            ret = self.send_aux(
                                req,
                                kv_chunk.prefill_aux_index,
                                target_rank_registration_info.dst_aux_ptrs,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status, local_rank
                                    )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def send_kvcahe_c8(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
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
                self.kv_args.dequant_scale_data_ptrs[layer_id],
                self.kv_args.dequant_scale_item_lens[layer_id],
            )
            for layer_id in range(num_layers)
        ]

        # Worker function for processing all layers in a batch
        transfer_blocks = []
        for (
            src_ptr,
            dst_ptr,
            item_len,
            dequant_scale_ptr,
            dequant_scale_item_len,
        ) in layers_params:
            tmp_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                dequant_scale_addr = (
                    dequant_scale_ptr + int(decode_index[0]) * dequant_scale_item_len
                )
                tmp_blocks.append((src_addr, dst_addr, length, dequant_scale_addr))
            transfer_blocks.extend(tmp_blocks)

        src_addrs, dst_addrs, lengths, dequant_scale_addrs = zip(*transfer_blocks)
        return self.engine.batch_transfer_write_with_quant(
            mooncake_session_id,
            list(src_addrs),
            list(dst_addrs),
            list(lengths),
            list(dequant_scale_addrs),
            [],
            self.kv_args.dequant_unit_num,
        )


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
