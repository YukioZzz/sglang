from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
from mori.cpp import TransferStatus
from mori.io import (
    BackendType,
    EngineDesc,
    IOEngine,
    IOEngineConfig,
    MemoryDesc,
    MemoryLocationType,
    PollCqMode,
    RdmaBackendConfig,
)

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import get_int_env_var
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)
MORI_GUARD = b"MoriMsgGuard"


def _pack_mem_desc_list(mems: List[MemoryDesc]) -> bytes:
    if not mems:
        return b""
    packed_descs = [mem.pack() for mem in mems]
    return msgspec.msgpack.encode(packed_descs)


def _unpack_mem_desc_list(blob: bytes) -> List[MemoryDesc]:
    if not blob:
        return []
    desc_blobs = msgspec.msgpack.decode(blob)
    return [MemoryDesc.unpack(b) for b in desc_blobs]


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    engine_key: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    is_dummy: bool
    # [PATCH-12-v2] dsv4 state plumbing — SWA-page indices receiver pre-allocated
    dst_state_indices: npt.NDArray[np.int32] = dataclasses.field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )

    @classmethod
    def from_zmq(cls, payload: List[bytes]) -> TransferInfo:
        room = int(payload[0].decode("ascii"))
        endpoint = payload[1].decode("ascii")
        dst_port = int(payload[2].decode("ascii"))
        engine_key = payload[3].decode("ascii")

        if payload[4]:
            dst_kv_indices = np.frombuffer(payload[4], dtype=np.int32)
        else:
            dst_kv_indices = np.array([], dtype=np.int32)

        if payload[5]:
            dst_aux_index = int(payload[5].decode("ascii"))
        else:
            dst_aux_index = -1

        # [PATCH-12-v2] decode state_indices from the long-reserved-but-empty
        # state_bytes slot at payload[6]. .copy() makes the buffer owned
        # so the original zmq frame can be released.
        if len(payload) > 6 and payload[6]:
            dst_state_indices = np.frombuffer(payload[6], dtype=np.int32).copy()
        else:
            dst_state_indices = np.array([], dtype=np.int32)

        required_dst_info_num = (
            int(payload[7].decode("ascii")) if len(payload) > 7 else 1
        )
        is_dummy = dst_kv_indices.size == 0 and dst_aux_index < 0
        return cls(
            room=room,
            endpoint=endpoint,
            dst_port=dst_port,
            engine_key=engine_key,
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=required_dst_info_num,
            is_dummy=is_dummy,
            dst_state_indices=dst_state_indices,  # [PATCH-12-v2]
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    endpoint: str
    dst_port: int
    engine_desc: EngineDesc
    dst_kv_mem_descs: List[MemoryDesc]
    dst_aux_mem_descs: List[MemoryDesc]
    dst_state_mem_descs: List[MemoryDesc]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int

    @property
    def engine_key(self) -> str:
        return self.engine_desc.key

    @classmethod
    def from_zmq(cls, payload: List[bytes]) -> KVArgsRegisterInfo:
        endpoint = payload[1].decode("ascii")
        dst_port = int(payload[2].decode("ascii"))
        engine_desc = EngineDesc.unpack(payload[3])
        dst_kv_mem_descs = _unpack_mem_desc_list(payload[4])
        dst_aux_mem_descs = _unpack_mem_desc_list(payload[5])
        dst_state_mem_descs = _unpack_mem_desc_list(payload[6])
        gpu_id = int(payload[7].decode("ascii"))
        decode_tp_size = int(payload[8].decode("ascii"))
        decode_tp_rank = int(payload[9].decode("ascii"))
        dst_kv_item_len = int(payload[10].decode("ascii"))
        return cls(
            endpoint=endpoint,
            dst_port=dst_port,
            engine_desc=engine_desc,
            dst_kv_mem_descs=dst_kv_mem_descs,
            dst_aux_mem_descs=dst_aux_mem_descs,
            dst_state_mem_descs=dst_state_mem_descs,
            gpu_id=gpu_id,
            decode_tp_size=decode_tp_size,
            decode_tp_rank=decode_tp_rank,
            dst_kv_item_len=dst_kv_item_len,
        )


class AuxDataCodec:
    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


@dataclasses.dataclass
class TPSliceConfig:
    page_size: int
    src_item_len: int
    dst_item_len: int
    bytes_per_token_src: int
    bytes_per_token_dst: int
    src_head_slice_offset: int
    dst_head_slice_offset: int
    heads_bytes_per_token_to_send: int


class MoriKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.engine = self._init_engine()
        self.engine_desc = self.engine.get_engine_desc()
        self.kv_mem_descs: List[MemoryDesc] = []
        self.aux_mem_descs: List[MemoryDesc] = []
        self.state_mem_descs: List[MemoryDesc] = []
        self.transfer_lock = threading.Lock()
        self._register_local_buffers()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.room_to_bootstrap_addr: Dict[int, str] = {}
            self._start_decode_thread()

    def _init_engine(self) -> IOEngine:
        if self.kv_args.ib_device:
            os.environ["MORI_RDMA_DEVICES"] = self.kv_args.ib_device

        self.local_ip = get_local_ip_auto()
        config = IOEngineConfig(host=self.local_ip, port=0)

        engine_key = (
            f"io-{self.disaggregation_mode.value}-"
            f"dp{self.system_dp_rank}-tp{self.attn_tp_rank}-"
            f"pid{os.getpid()}-{self.local_ip}"
        )

        engine = IOEngine(engine_key, config)
        poll_mode = PollCqMode.POLLING

        # Number of RDMA Queue Pairs (QPs) used per transfer operation.
        # Higher values can increase parallelism and bandwidth utilization.
        # Default: 1
        qp_per_transfer = get_int_env_var("SGLANG_MORI_QP_PER_TRANSFER", 1)

        # Number of RDMA work requests posted in a single batch to each QP.
        # Larger batch sizes reduce per-operation overhead and improve throughput
        # at the cost of higher latency. Use -1 for automatic sizing based on
        # the number of merged work requests and available endpoints.
        # Default: -1 (automatic)
        post_batch_size = get_int_env_var("SGLANG_MORI_POST_BATCH_SIZE", -1)

        # Number of worker threads in the RDMA executor thread pool.
        # Each worker handles RDMA operations on a separate CPU core (with affinity).
        # More workers can improve parallelism for large batch transfers across
        # multiple QPs, but excessive threads may cause contention.
        # Default: 1
        num_worker_threads = get_int_env_var("SGLANG_MORI_NUM_WORKERS", 1)

        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer,
            post_batch_size,
            num_worker_threads,
            poll_mode,
            False,
        )
        engine.create_backend(BackendType.RDMA, rdma_cfg)
        actual_port = engine.get_engine_desc().port
        assert actual_port > 0, f"Failed to bind port for engine {engine_key}"
        logger.debug(
            "Initialized Mori IOEngine %s at %s:%s (qp_per_transfer=%s, workers=%s, poll_mode=%s)",
            engine_key,
            self.local_ip,
            actual_port,
            qp_per_transfer,
            num_worker_threads,
            poll_mode.name,
        )
        return engine

    def _register_local_buffers(self) -> None:
        for ptr, length in zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens):
            mem_desc = self.engine.register_memory(
                ptr,
                length,
                self.kv_args.gpu_id,
                MemoryLocationType.GPU,
            )
            self.kv_mem_descs.append(mem_desc)
        for ptr, length in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens):
            desc = self.engine.register_memory(
                ptr,
                length,
                -1,
                MemoryLocationType.CPU,
            )
            self.aux_mem_descs.append(desc)
        for ptr, length in zip(
            self.kv_args.state_data_ptrs, getattr(self.kv_args, "state_data_lens", [])
        ):
            desc = self.engine.register_memory(
                ptr,
                length,
                self.kv_args.gpu_id,
                MemoryLocationType.GPU,
            )
            self.state_mem_descs.append(desc)

    def _handle_register_message(self, payload: List[bytes]) -> None:
        try:
            register_info = KVArgsRegisterInfo.from_zmq(payload)
            self._add_remote_peer(register_info)
        except Exception:
            logger.exception("Failed to register remote peer")

    def _handle_transfer_message(self, payload: List[bytes]) -> None:
        try:
            transfer_info = TransferInfo.from_zmq(payload)
            infos = self.transfer_infos.setdefault(transfer_info.room, {})
            infos[transfer_info.engine_key] = transfer_info

            if len(infos) >= transfer_info.required_dst_info_num:
                logger.debug(
                    "Bootstrap room %s got enough transfer info (%s)",
                    transfer_info.room,
                    len(infos),
                )
                self.update_status(transfer_info.room, KVPoll.WaitingForInput)
        except Exception:
            logger.exception("Failed to parse transfer info message")

    def _validate_message(self, msg: List[bytes]) -> Optional[List[bytes]]:
        if not msg or msg[0] != MORI_GUARD:
            logger.warning("Received malformed bootstrap message")
            return None
        payload = msg[1:]
        if not payload:
            return None
        return payload

    def _start_bootstrap_thread(self) -> None:
        def bootstrap_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    payload = self._validate_message(msg)
                    if payload is None:
                        continue
                    room = payload[0].decode("ascii")

                    if room == "None":
                        self._handle_register_message(payload)
                    else:
                        self._handle_transfer_message(payload)
                except Exception:
                    logger.exception("Bootstrap worker failed")

        threading.Thread(target=bootstrap_worker, daemon=True).start()

    def _cleanup_room_tracking(self, bootstrap_room: int) -> None:
        bootstrap_addr = self.room_to_bootstrap_addr.pop(bootstrap_room, None)
        if bootstrap_addr is not None:
            rooms = self.addr_to_rooms_tracker.get(bootstrap_addr)
            if rooms is not None:
                rooms.discard(bootstrap_room)
                if not rooms:
                    self.addr_to_rooms_tracker.pop(bootstrap_addr, None)

    def _start_decode_thread(self) -> None:
        def decode_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    if msg and msg[0] == MoriKVManager.AUX_DATA_HEADER:
                        self._handle_aux_data(msg)
                        continue

                    if not msg or msg[0] != MORI_GUARD:
                        logger.warning(
                            "Received malformed status message on decode worker"
                        )
                        continue
                    payload = msg[1:]
                    if len(payload) < 3:
                        logger.warning("Incomplete status payload received")
                        continue
                    bootstrap_room = int(payload[0].decode("ascii"))
                    status_code = int(payload[1].decode("ascii"))
                    prefill_rank = int(payload[2].decode("ascii"))
                    failure_reason = (
                        payload[3].decode("utf-8")
                        if len(payload) > 3 and payload[3]
                        else None
                    )

                    if status_code == KVPoll.Success:
                        tracker = self.prefill_response_tracker[bootstrap_room]
                        tracker.add(prefill_rank)
                        expected = self.required_prefill_response_num_table.get(
                            bootstrap_room, 1
                        )
                        if len(tracker) >= expected:
                            self.prefill_response_tracker.pop(bootstrap_room, None)
                            self.update_status(bootstrap_room, KVPoll.Success)
                            self._cleanup_room_tracking(bootstrap_room)
                    elif status_code == KVPoll.Failed:
                        if failure_reason:
                            self.record_failure(bootstrap_room, failure_reason)
                        self.prefill_response_tracker.pop(bootstrap_room, None)
                        self.update_status(bootstrap_room, KVPoll.Failed)
                        self._cleanup_room_tracking(bootstrap_room)
                    else:
                        logger.warning(
                            "Unknown status code %s received for room %s",
                            status_code,
                            bootstrap_room,
                        )
                except Exception:
                    logger.exception("Decode status worker failed")

        threading.Thread(target=decode_worker, daemon=True).start()

    def notify_decode_status(
        self,
        infos: List[TransferInfo],
        bootstrap_room: int,
        status: KVPoll,
        failure_reason: Optional[str] = None,
    ) -> None:
        if not infos:
            return
        payload = [
            MORI_GUARD,
            str(bootstrap_room).encode("ascii"),
            str(int(status)).encode("ascii"),
            str(self.attn_tp_rank * self.pp_size + self.pp_rank).encode("ascii"),
            failure_reason.encode("utf-8") if failure_reason else b"",
        ]
        for info in infos:
            try:
                na = NetworkAddress(info.endpoint, info.dst_port)
                socket = self._connect(na.to_tcp(), is_ipv6=na.is_ipv6)
                socket.send_multipart(payload)
            except Exception:
                logger.exception(
                    "Failed to sync status %s to decode endpoint %s:%s for room %s",
                    status,
                    info.endpoint,
                    info.dst_port,
                    bootstrap_room,
                )

    def _add_remote_peer(self, register_info: KVArgsRegisterInfo) -> None:
        engine_key = register_info.engine_key
        if engine_key in self.decode_kv_args_table:
            logger.debug("Remote peer %s already registered. Skipping.", engine_key)
            return
        self.engine.register_remote_engine(register_info.engine_desc)
        self.decode_kv_args_table[engine_key] = register_info
        logger.debug(
            "Registered decode peer %s (%s:%s)",
            engine_key,
            register_info.endpoint,
            register_info.dst_port,
        )

    def _get_mha_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[
        List[MemoryDesc], List[MemoryDesc], List[MemoryDesc], List[MemoryDesc], int
    ]:
        src_descs = self.kv_mem_descs
        if not src_descs:
            raise RuntimeError("KV memory descriptors are empty on prefill side")

        num_local_layers = len(src_descs) // 2
        src_k_descs = src_descs[:num_local_layers]
        src_v_descs = src_descs[num_local_layers:]

        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        dst_total_layers = len(dst_mem_descs) // 2
        if len(dst_mem_descs) < 2 or end_layer > dst_total_layers:
            raise ValueError(
                "Destination KV descriptors do not match prefill pp configuration"
            )
        dst_k_descs = dst_mem_descs[start_layer:end_layer]
        dst_v_descs = dst_mem_descs[
            dst_total_layers + start_layer : dst_total_layers + end_layer
        ]
        return src_k_descs, src_v_descs, dst_k_descs, dst_v_descs, num_local_layers

    def _get_mla_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[List[MemoryDesc], List[MemoryDesc], int]:
        src_descs = self.kv_mem_descs
        num_local_layers = len(src_descs)
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        if end_layer > len(dst_mem_descs):
            raise ValueError(
                "Destination MLA KV descriptors do not match prefill pp configuration"
            )
        dst_slice = dst_mem_descs[start_layer:end_layer]
        return src_descs, dst_slice, num_local_layers

    def _issue_layer_transfers(
        self,
        src_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        kv_item_len: int,
        src_groups: List[List[int]],
        dst_groups: List[List[int]],
    ) -> List[TransferStatus]:
        # [PATCH-LAYER-SWA] dsv4 layer swa-wrap + diag
        # Detect SWA-sized layers by comparing this layer's src_desc.size
        # against the largest layer MR size cached on first call. For SWA
        # layers wrap slot ids via modulo with min(src_cap, dst_cap) so
        # both sides land at the SAME relative SWA slot; emit singleton
        # WRs (one per slot) since modulo breaks contiguity assumptions.
        # Non-SWA layers go through the original fast path.
        if not src_groups:
            return []

        try:
            _layer_diag_bw = int(os.environ.get("DSV4_LAYER_DIAG_BATCHWRITE", "0"))
        except Exception:
            _layer_diag_bw = 0
        try:
            _layer_diag_failed = int(os.environ.get("DSV4_LAYER_DIAG_FAILED", "1"))
        except Exception:
            _layer_diag_failed = 1
        try:
            _layer_diag_every = int(os.environ.get("DSV4_LAYER_DIAG_EVERY", "100"))
        except Exception:
            _layer_diag_every = 100
        if _layer_diag_every <= 0:
            _layer_diag_every = 1
        try:
            _swa_wrap_enabled = int(os.environ.get("DSV4_LAYER_SWA_WRAP", "1"))
        except Exception:
            _swa_wrap_enabled = 1

        _layer_call_idx = getattr(self, "_layer_diag_call_idx", 0)
        self._layer_diag_call_idx = _layer_call_idx + 1

        # Lazily cache the largest local kv_mem_desc size so we can
        # classify per-layer descs as SWA vs full at runtime.
        _ref_size = getattr(self, "_max_kv_mem_desc_size", None)
        if _ref_size is None:
            try:
                _ref_size = max(
                    (int(getattr(d, "size", 0) or 0)
                     for d in getattr(self, "kv_mem_descs", []) or []),
                    default=0,
                )
            except Exception:
                _ref_size = 0
            self._max_kv_mem_desc_size = _ref_size

        try:
            _src_size = int(getattr(src_desc, "size", 0) or 0)
            _dst_size = int(getattr(dst_desc, "size", 0) or 0)
            _src_id = int(getattr(src_desc, "id", -1) or -1)
            _dst_id = int(getattr(dst_desc, "id", -1) or -1)
        except Exception:
            _src_size = _dst_size = _src_id = _dst_id = -1

        _is_swa_layer = bool(
            _swa_wrap_enabled and _ref_size > 0
            and _src_size > 0 and _src_size < _ref_size
            and kv_item_len > 0
        )

        if _is_swa_layer:
            # Wrap into common SWA slot space.
            _src_cap = _src_size // kv_item_len
            _dst_cap = _dst_size // kv_item_len if _dst_size > 0 else _src_cap
            _swa_cap = min(_src_cap, _dst_cap)
            if _swa_cap <= 0:
                # Degenerate: fall back to original path to keep behavior.
                _is_swa_layer = False

        if _is_swa_layer:
            # Singleton WRs, modulo-wrapped both sides.
            local_offsets = []
            remote_offsets = []
            sizes = []
            _src_max_full = -1
            _dst_max_full = -1
            for _src_grp, _dst_grp in zip(src_groups, dst_groups):
                for _s, _d in zip(_src_grp, _dst_grp):
                    _s_int = int(_s)
                    _d_int = int(_d)
                    if _s_int > _src_max_full:
                        _src_max_full = _s_int
                    if _d_int > _dst_max_full:
                        _dst_max_full = _d_int
                    local_offsets.append((_s_int % _swa_cap) * kv_item_len)
                    remote_offsets.append((_d_int % _swa_cap) * kv_item_len)
                    sizes.append(kv_item_len)
            if not sizes:
                return []
        else:
            # Original (full-pool) path — unchanged from upstream.
            local_offsets = [int(src_group[0]) * kv_item_len for src_group in src_groups]
            remote_offsets = [int(dst_group[0]) * kv_item_len for dst_group in dst_groups]
            sizes = [len(src_group) * kv_item_len for src_group in src_groups]

        # Permanent pre-flight bounds assertion: mirror MoRI's C++ check
        # (common.cpp:334) BEFORE we hand WRs to the engine. With the SWA
        # wrap above + identity full_to_swa mapping (patch 6b), this should
        # never trip on healthy traffic; any True->False transition is a
        # regression and aborts loudly instead of producing 8000+ MoRI
        # [io][error] entries from the bounds-violating WRs.
        # Knob DSV4_LAYER_PREFLIGHT_STRICT=0 downgrades to a warning.
        try:
            _pre_local_max = max(
                (lo + sz for lo, sz in zip(local_offsets, sizes)), default=0
            )
            _pre_remote_max = max(
                (ro + sz for ro, sz in zip(remote_offsets, sizes)), default=0
            )
        except Exception:
            _pre_local_max = _pre_remote_max = -1
        _pre_pass = (
            _pre_local_max <= _src_size and _pre_remote_max <= _dst_size
        ) if (_src_size > 0 and _dst_size > 0) else True
        if not _pre_pass:
            try:
                _strict = int(os.environ.get("DSV4_LAYER_PREFLIGHT_STRICT", "1"))
            except Exception:
                _strict = 1
            _err_msg = (
                "[PATCH-LAYER-SWA PREFLIGHT] OOB: "
                "src_id=%d dst_id=%d src_size=%d dst_size=%d swa=%s "
                "local_max=%d remote_max=%d n_wrs=%d kv_item_len=%d"
            ) % (
                _src_id, _dst_id, _src_size, _dst_size, _is_swa_layer,
                _pre_local_max, _pre_remote_max, len(sizes), kv_item_len,
            )
            if _strict:
                raise RuntimeError(_err_msg)
            logger.error(_err_msg)

        transfer_uid = self.engine.allocate_transfer_uid()

        statuses = self.engine.batch_write(
            [src_desc],
            [local_offsets],
            [dst_desc],
            [remote_offsets],
            [sizes],
            [transfer_uid],
        )

        if not (_layer_diag_bw or _layer_diag_failed) or not statuses:
            return statuses

        def _is_ok(_st):
            # Mirror of DIAG-V4 detection in _transfer_state_buffers; cached.
            _det = getattr(self, "_p12v2_status_detect", None)
            if _det is None:
                try:
                    if callable(getattr(_st, 'Failed', None)):
                        self._p12v2_status_detect = 'not_Failed()'
                        return not bool(_st.Failed())
                    if callable(getattr(_st, 'Succeeded', None)):
                        self._p12v2_status_detect = 'Succeeded()'
                        return bool(_st.Succeeded())
                except Exception:
                    return True
                self._p12v2_status_detect = 'fallback_assume_ok'
                return True
            try:
                if _det == 'not_Failed()':
                    return not bool(_st.Failed())
                if _det == 'Succeeded()':
                    return bool(_st.Succeeded())
                return True
            except Exception:
                return True

        try:
            _n_ok = sum(1 for _s in statuses if _is_ok(_s))
        except Exception:
            _n_ok = -1
        _n_bad = (len(statuses) - _n_ok) if _n_ok >= 0 else -1

        try:
            _local_max = max((lo + sz for lo, sz in zip(local_offsets, sizes)), default=0)
            _remote_max = max((ro + sz for ro, sz in zip(remote_offsets, sizes)), default=0)
        except Exception:
            _local_max = _remote_max = -1
        _python_check_pass = (
            _local_max <= _src_size and _remote_max <= _dst_size
        ) if (_src_size > 0 and _dst_size > 0) else True

        if _layer_diag_bw and (_layer_call_idx % _layer_diag_every == 0):
            try:
                logger.warning(
                    "[PATCH-LAYER-SWA BW] call#%d uid=%d src_id=%d dst_id=%d "
                    "src_size=%d dst_size=%d swa=%s n_wrs=%d ok=%d bad=%d "
                    "local_max=%d remote_max=%d py_pass=%s",
                    _layer_call_idx, int(transfer_uid), _src_id, _dst_id,
                    _src_size, _dst_size, _is_swa_layer, len(sizes), _n_ok, _n_bad,
                    _local_max, _remote_max, _python_check_pass,
                )
            except Exception as _e:
                logger.warning("[PATCH-LAYER-SWA BW] log error: %s", _e)

        if _layer_diag_failed and _n_bad and _n_bad > 0:
            try:
                _code = -1
                _msg = ""
                for _st in statuses:
                    if not _is_ok(_st):
                        try:
                            _code = int(_st.Code())
                            _msg = str(_st.Message())[:80]
                        except Exception:
                            pass
                        break
                logger.warning(
                    "[PATCH-LAYER-SWA FAIL] call#%d uid=%d src_id=%d dst_id=%d "
                    "src_size=%d dst_size=%d swa=%s n_wrs=%d bad=%d/%d "
                    "local_max=%d remote_max=%d py_pass=%s code=%d msg=%r "
                    "local_offs_head=%s remote_offs_head=%s sizes_head=%s",
                    _layer_call_idx, int(transfer_uid), _src_id, _dst_id,
                    _src_size, _dst_size, _is_swa_layer, len(sizes), _n_bad, len(statuses),
                    _local_max, _remote_max, _python_check_pass, _code, _msg,
                    list(local_offsets[:4]), list(remote_offsets[:4]),
                    list(sizes[:4]),
                )
            except Exception as _e:
                logger.warning("[PATCH-LAYER-SWA FAIL] log error: %s", _e)

        return statuses

    def _build_tp_slice_config(self, peer_info: KVArgsRegisterInfo) -> TPSliceConfig:
        page_size = self.kv_args.page_size

        src_item_len = self.kv_args.kv_item_lens[0]
        dst_item_len = peer_info.dst_kv_item_len

        bytes_per_token_src = src_item_len // page_size
        bytes_per_token_dst = dst_item_len // page_size

        prefill_tp_size = self.attn_tp_size
        decode_tp_size = peer_info.decode_tp_size

        num_kv_heads = self.kv_args.kv_head_num
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * prefill_tp_size // decode_tp_size
        if dst_heads_per_rank == 0:
            raise ValueError("Destination heads per rank evaluates to zero")

        bytes_per_head_slice = bytes_per_token_dst // dst_heads_per_rank
        if bytes_per_head_slice == 0:
            raise ValueError("Head slice size evaluates to zero")

        local_tp_rank = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank = peer_info.decode_tp_rank % decode_tp_size

        if prefill_tp_size > decode_tp_size:
            src_head_start = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start = local_tp_rank * src_heads_per_rank
        else:
            src_head_start = (dst_tp_rank * dst_heads_per_rank) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start = 0

        src_head_slice_offset = src_head_start * bytes_per_head_slice
        dst_head_slice_offset = dst_head_start * bytes_per_head_slice
        heads_bytes_per_token = num_heads_to_send * bytes_per_head_slice

        if heads_bytes_per_token > bytes_per_token_dst:
            raise ValueError(
                "Slice size exceeds destination token capacity for TP slice transfer"
            )

        return TPSliceConfig(
            page_size=page_size,
            src_item_len=src_item_len,
            dst_item_len=dst_item_len,
            bytes_per_token_src=bytes_per_token_src,
            bytes_per_token_dst=bytes_per_token_dst,
            src_head_slice_offset=src_head_slice_offset,
            dst_head_slice_offset=dst_head_slice_offset,
            heads_bytes_per_token_to_send=heads_bytes_per_token,
        )

    def _issue_tp_slice_transfers(
        self,
        src_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        kv_indices: npt.NDArray[np.int32],
        dst_indices: npt.NDArray[np.int32],
        tp_cfg: TPSliceConfig,
    ) -> List[TransferStatus]:
        if kv_indices.size == 0 or dst_indices.size == 0:
            return []

        limit = min(kv_indices.size, dst_indices.size)
        if not limit:
            return []

        src_pages = kv_indices[:limit].astype(np.int64)
        dst_pages = dst_indices[:limit].astype(np.int64)
        token_slots = np.arange(tp_cfg.page_size, dtype=np.int64)

        src_page_bases = src_pages * tp_cfg.src_item_len
        dst_page_bases = dst_pages * tp_cfg.dst_item_len

        src_token_offsets = token_slots * tp_cfg.bytes_per_token_src
        dst_token_offsets = token_slots * tp_cfg.bytes_per_token_dst

        local_offsets = (
            (
                src_page_bases[:, np.newaxis]
                + src_token_offsets
                + tp_cfg.src_head_slice_offset
            )
            .flatten()
            .tolist()
        )
        remote_offsets = (
            (
                dst_page_bases[:, np.newaxis]
                + dst_token_offsets
                + tp_cfg.dst_head_slice_offset
            )
            .flatten()
            .tolist()
        )

        num_transfers = limit * tp_cfg.page_size
        sizes = [tp_cfg.heads_bytes_per_token_to_send] * num_transfers

        if not local_offsets:
            return []

        transfer_uid = self.engine.allocate_transfer_uid()
        statuses = self.engine.batch_write(
            [src_desc],
            [local_offsets],
            [dst_desc],
            [remote_offsets],
            [sizes],
            [transfer_uid],
        )
        return statuses

    def send_kvcache(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
    ) -> List[TransferStatus]:
        src_groups, dst_groups = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )
        statuses = []
        kv_item_len = self.kv_args.kv_item_lens[0]
        if self.is_mla_backend:
            (
                src_descs,
                dst_descs,
                layers_current_pp_stage,
            ) = self._get_mla_mem_desc_slices(peer_info.dst_kv_mem_descs)
            for layer_id in range(layers_current_pp_stage):
                statuses.extend(
                    self._issue_layer_transfers(
                        src_descs[layer_id],
                        dst_descs[layer_id],
                        kv_item_len,
                        src_groups,
                        dst_groups,
                    )
                )
        else:
            tp_mismatch = peer_info.decode_tp_size != self.attn_tp_size
            (
                src_k_descs,
                src_v_descs,
                dst_k_descs,
                dst_v_descs,
                layers_current_pp_stage,
            ) = self._get_mha_mem_desc_slices(peer_info.dst_kv_mem_descs)

            if tp_mismatch:
                tp_cfg = self._build_tp_slice_config(peer_info)
                for layer_id in range(layers_current_pp_stage):
                    statuses.extend(
                        self._issue_tp_slice_transfers(
                            src_k_descs[layer_id],
                            dst_k_descs[layer_id],
                            prefill_kv_indices,
                            dst_kv_indices,
                            tp_cfg,
                        )
                    )
                    statuses.extend(
                        self._issue_tp_slice_transfers(
                            src_v_descs[layer_id],
                            dst_v_descs[layer_id],
                            prefill_kv_indices,
                            dst_kv_indices,
                            tp_cfg,
                        )
                    )
            else:
                src_groups, dst_groups = group_concurrent_contiguous(
                    prefill_kv_indices, dst_kv_indices
                )
                for layer_id in range(layers_current_pp_stage):
                    statuses.extend(
                        self._issue_layer_transfers(
                            src_k_descs[layer_id],
                            dst_k_descs[layer_id],
                            kv_item_len,
                            src_groups,
                            dst_groups,
                        )
                    )
                    statuses.extend(
                        self._issue_layer_transfers(
                            src_v_descs[layer_id],
                            dst_v_descs[layer_id],
                            kv_item_len,
                            src_groups,
                            dst_groups,
                        )
                    )

        return statuses

    def send_aux(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        return self.send_aux_tcp(peer_info, prefill_aux_index, dst_aux_index, room)

    def send_aux_tcp(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=peer_info.endpoint,
                dst_port=peer_info.dst_port,
                room=room,
                buffer_index=i,
                aux_index=dst_aux_index,
                data=data,
            )

        return []

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        na = NetworkAddress(remote, dst_port)
        socket = self._connect(na.to_tcp(), is_ipv6=na.is_ipv6)

        socket.send_multipart(
            [
                MoriKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ]
        )

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread."""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    # [PATCH-12-v2] dsv4 sparse state-buffer transfer ------------------
    def _transfer_state_buffers(
        self,
        peer_info: "KVArgsRegisterInfo",
        src_state_indices: npt.NDArray[np.int32],
        dst_state_indices: npt.NDArray[np.int32],
    ) -> List[TransferStatus]:
        """Per-segment, page-indexed transfer of DSv4 state pools
        (compress + indexer compress).  Both index arrays must be in
        SWA-page space — prefill.py / decode.py [PATCH-13] is
        responsible for translating full-pool indices via
        ``translate_loc_from_full_to_swa`` before calling."""
        statuses: List[TransferStatus] = []
        try:
            src_descs = list(getattr(self, "state_mem_descs", None) or [])
            dst_descs = list(getattr(peer_info, "dst_state_mem_descs", None) or [])
            item_lens = list(getattr(self.kv_args, "state_item_lens", None) or [])
        except Exception:
            return statuses
        if not src_descs or not dst_descs or not item_lens:
            return statuses
        # [PATCH-12-v2 / Option-3] singleton groups (one page per WR).
        # Rationale: MoRI's per-WR message-length limit is exceeded when
        # state transfers use contiguous groups (e.g. [4,5,6,7] -> a
        # single WR of 4*item_len bytes; with item_len=524288 for c128
        # that is ~2 MB per WR, which MoRI rejects with 'message length
        # out of range'). v2 (all-zeros mapping) accidentally worked
        # because [0,0,0,0] is not arithmetically contiguous and
        # group_concurrent_contiguous emitted 4 singleton groups.
        # Option-3 explicitly keeps singletons regardless of index
        # contiguity. WR count grows to len(state_indices) per segment,
        # but each WR stays at 1*item_len bytes (same shape MoRI's main
        # KV channel uses successfully on non-DSv4 models).
        src_list = [int(p) for p in src_state_indices.tolist()]
        dst_list = [int(p) for p in dst_state_indices.tolist()]
        if len(src_list) != len(dst_list):
            logger.warning(
                "[PATCH-12-v2] state_indices length mismatch: "
                "src=%d dst=%d; truncating to shorter",
                len(src_list), len(dst_list),
            )
            _m = min(len(src_list), len(dst_list))
            src_list, dst_list = src_list[:_m], dst_list[:_m]
        if not src_list:
            return statuses
        # [PATCH-12-v2 / DIAG-V3] per-request counter (replaces _p12v2_logged
        # one-shot flag). Set DSV4_STATE_DIAG_EVERY=N to log every Nth call;
        # call #0 is always logged. We deliberately keep this OUTSIDE the
        # lock-guarded region so periodic diag fires even when lock is on.
        _call_idx = int(getattr(self, "_p12v2_call_idx", 0))
        self._p12v2_call_idx = _call_idx + 1
        try:
            _diag_every = int(os.environ.get("DSV4_STATE_DIAG_EVERY", "0"))
        except Exception:
            _diag_every = 0
        _should_diag = (
            _call_idx == 0
            or (_diag_every > 0 and (_call_idx % _diag_every) == 0)
        )
        _did_log = not _should_diag
        # [PATCH-12-v2 / Option-3 size gate] Optional cap to skip segments
        # whose per-page (item_len) bytes exceed the MoRI/IB per-WR limit.
        # Setting DSV4_STATE_MAX_ITEM_LEN=524288 effectively excludes c128
        # indexer compress-state segments (il=1 MB) while keeping SWA KV
        # (146 KB), c4 compress state (128 KB) and the 32 KB tail segments
        # included. Used to A/B test whether MoRI's 'message length out of
        # range' is driven by per-WR size vs MR-layout/offset constraints.
        try:
            _max_il = int(os.environ.get("DSV4_STATE_MAX_ITEM_LEN", "0"))
        except Exception:
            _max_il = 0
        try:
            _chunk_cap = int(os.environ.get("DSV4_STATE_CHUNK_CAP_BYTES", "0"))
        except Exception:
            _chunk_cap = 0
        try:
            _agg_call = int(os.environ.get("DSV4_STATE_AGGREGATE_BATCH", "1"))
        except Exception:
            _agg_call = 1
        _skipped_total = 0
        _kept_total = 0
        _chunked_total = 0
        # [PATCH-12-v2 / Option-3 aggregation] Accumulate per-segment
        # offset/size lists into outer lists, then issue ONE batch_write
        # for the whole request instead of 152 calls. MoRI's batch_write
        # accepts (List[MemoryDesc], List[List[offset]], ...) so multiple
        # descriptors can be submitted in a single engine call.  Conc>1
        # failure was traced to 4 concurrent requests issuing ~608
        # batch_write calls in parallel across 4 MoRI IO threads which
        # overflowed something internal. One call per request keeps the
        # call frequency identical to upstream send_kvcache (one call per
        # layer there — same order of magnitude as one per request here).
        agg_src_descs = []
        agg_dst_descs = []
        agg_local_offsets = []
        agg_remote_offsets = []
        agg_sizes = []
        agg_xuids = []
        agg_seg_idx = []
        n = min(len(src_descs), len(dst_descs), len(item_lens))
        for idx in range(n):
            item_len = int(item_lens[idx])
            if item_len <= 0:
                continue
            if _max_il > 0 and item_len > _max_il:
                _skipped_total += 1
                continue
            _kept_total += 1
            if (not _did_log) and idx == 0:
                src_total = int(getattr(src_descs[idx], "size", 0) or 0)
                dst_total = int(getattr(dst_descs[idx], "size", 0) or 0)
                logger.warning(
                    "[PATCH-12-v2 DIAG-V3 call#%d seg0] item_len=%d "
                    "npages=%d src_total=%d dst_total=%d "
                    "max_src_page=%d max_dst_page=%d",
                    _call_idx, item_len, len(src_list), src_total, dst_total,
                    int(src_state_indices.max()) if src_state_indices.size else -1,
                    int(dst_state_indices.max()) if dst_state_indices.size else -1,
                )
            # Build (local_offsets, remote_offsets, sizes) lists for THIS
            # segment. Chunk when item_len exceeds the per-WR cap.
            _local_off: List[int] = []
            _remote_off: List[int] = []
            _sizes: List[int] = []
            if _chunk_cap > 0 and item_len > _chunk_cap:
                _chunked_total += 1
                for _sp, _dp in zip(src_list, dst_list):
                    _base_src = _sp * item_len
                    _base_dst = _dp * item_len
                    _remaining = item_len
                    _off = 0
                    while _remaining > 0:
                        _sz = _chunk_cap if _remaining > _chunk_cap else _remaining
                        _local_off.append(_base_src + _off)
                        _remote_off.append(_base_dst + _off)
                        _sizes.append(_sz)
                        _off += _sz
                        _remaining -= _sz
            else:
                for _sp, _dp in zip(src_list, dst_list):
                    _local_off.append(_sp * item_len)
                    _remote_off.append(_dp * item_len)
                    _sizes.append(item_len)
            agg_src_descs.append(src_descs[idx])
            agg_dst_descs.append(dst_descs[idx])
            agg_local_offsets.append(_local_off)
            agg_remote_offsets.append(_remote_off)
            agg_sizes.append(_sizes)
            agg_xuids.append(self.engine.allocate_transfer_uid())
            agg_seg_idx.append(idx)
        # [PATCH-12-v2 / DIAG-V3] optional lock acquisition to serialize
        # batch_write across MoRI sender threads. DSV4_STATE_SERIALIZE=1.
        try:
            _do_lock = int(os.environ.get("DSV4_STATE_SERIALIZE", "0"))
        except Exception:
            _do_lock = 0
        _lock_p12v2 = None
        if _do_lock and agg_src_descs:
            _lock_p12v2 = getattr(self, "_p12v2_state_lock", None)
            if _lock_p12v2 is None:
                import threading as _threading_p12v2
                _lock_p12v2 = _threading_p12v2.Lock()
                self._p12v2_state_lock = _lock_p12v2
        _batch_statuses: list = []
        if _lock_p12v2 is not None:
            _lock_p12v2.acquire()
        try:
            if agg_src_descs:
                if _agg_call:
                    _batch_statuses = list(self.engine.batch_write(
                        agg_src_descs, agg_local_offsets,
                        agg_dst_descs, agg_remote_offsets,
                        agg_sizes, agg_xuids,
                    ))
                else:
                    for _i in range(len(agg_src_descs)):
                        _batch_statuses.extend(self.engine.batch_write(
                            [agg_src_descs[_i]], [agg_local_offsets[_i]],
                            [agg_dst_descs[_i]], [agg_remote_offsets[_i]],
                            [agg_sizes[_i]], [agg_xuids[_i]],
                        ))
                statuses.extend(_batch_statuses)
        finally:
            if _lock_p12v2 is not None:
                _lock_p12v2.release()
        if not _did_log:
            logger.warning(
                "[PATCH-12-v2 DIAG-V3 call#%d] kept=%d skipped=%d "
                "chunked=%d agg=%d lock=%d total_descs=%d total_WRs=%d "
                "cap=%d max_il=%d",
                _call_idx, _kept_total, _skipped_total, _chunked_total,
                _agg_call, _do_lock, len(agg_src_descs),
                sum(len(s) for s in agg_sizes),
                _chunk_cap, _max_il,
            )
        # [PATCH-12-v2 / DIAG-V4] failure dump using REAL TransferStatus API.
        # DIAG-V3's repr-substring fallback ('OK' / 'SUCCESS' in repr) was a
        # false-positive trap because pybind11's TransferStatus has no
        # __repr__ override -- repr is '<libmori_pybinds.TransferStatus
        # object at 0x...>', which contains 'OBJECT' but not the literals
        # we matched on, so EVERY status was flagged 'bad'. V4 introspects
        # the first status with dir() and picks the right detection method
        # (is_ok / ok / code / int / success), caches it on self, and uses
        # it for all subsequent calls. The chosen method is logged so we
        # can verify by hand. DSV4_STATE_DIAG_FAILED=0 to silence.
        # DSV4_STATE_DIAG_BATCHWRITE=1 enables a per-call BW summary log
        # (ok/bad count + uid range) regardless of failure presence -- used
        # to correlate our submissions with MoRI's [io][error] log stream
        # by timestamp.
        try:
            _diag_failed = int(os.environ.get("DSV4_STATE_DIAG_FAILED", "1"))
        except Exception:
            _diag_failed = 1
        try:
            _diag_bw = int(os.environ.get("DSV4_STATE_DIAG_BATCHWRITE", "0"))
        except Exception:
            _diag_bw = 0

        def _p12v2_is_ok(_status):
            # Real libmori_pybinds.TransferStatus API (probed on-cluster):
            #   .Code()       -> StatusCode enum (INIT=1, ...)
            #   .Message()    -> str (empty when no error)
            #   .Init()       -> bool (True freshly constructed)
            #   .InProgress() -> bool
            #   .Succeeded()  -> bool
            #   .Failed()     -> bool   <-- True iff actually broken
            #   .Wait()       -> blocks until terminal state
            # Right after batch_write returns, statuses are mostly
            # InProgress (async submitted) plus any pre-flight failures
            # which immediately go to Failed(). MoRI's [io][error] for
            # 'message length out of range' is the SYNCHRONOUS pre-flight
            # bounds check -- the bad WR's status will already be Failed()
            # when batch_write returns. So 'not Failed()' is the right
            # check for our diagnostic.
            _det = getattr(self, "_p12v2_status_detect", None)
            if _det is None:
                _members = [m for m in dir(_status) if not m.startswith('_')]
                _det = 'fallback_assume_ok'
                _ok = True
                # Probe runtime values for diagnostic ground-truth dump.
                _runtime_vals = {}
                for _name in ('Code', 'Message', 'Init', 'InProgress',
                              'Succeeded', 'Failed'):
                    try:
                        _fn = getattr(_status, _name, None)
                        if callable(_fn):
                            _runtime_vals[_name + '()'] = repr(_fn())[:80]
                    except Exception as _e:
                        _runtime_vals[_name + '()'] = 'ERR ' + type(_e).__name__
                # Detection priority: not Failed() (best -- only flags real
                # failures), then Succeeded() (treats InProgress as bad,
                # which would flood; only use if Failed missing), then the
                # generic fallbacks we kept from earlier.
                _fn = getattr(_status, 'Failed', None)
                if callable(_fn):
                    try:
                        _ok = not bool(_fn())
                        _det = 'not_Failed()'
                    except Exception:
                        pass
                if _det == 'fallback_assume_ok':
                    _fn = getattr(_status, 'Succeeded', None)
                    if callable(_fn):
                        try:
                            _ok = bool(_fn())
                            _det = 'Succeeded()'
                        except Exception:
                            pass
                if _det == 'fallback_assume_ok':
                    for _name in ('is_ok', 'ok'):
                        _fn = getattr(_status, _name, None)
                        if callable(_fn):
                            try:
                                _ok = bool(_fn())
                                _det = _name + '()'
                                break
                            except Exception:
                                continue
                self._p12v2_status_detect = _det
                logger.warning(
                    "[PATCH-12-v2 DIAG-V4 STATUS-API] type=%s repr=%r "
                    "members=%s runtime=%s chosen=%s first_ok=%s",
                    type(_status).__name__, repr(_status), _members,
                    _runtime_vals, _det, _ok,
                )
                return _ok
            try:
                if _det == 'not_Failed()':
                    return not bool(_status.Failed())
                if _det == 'Succeeded()':
                    return bool(_status.Succeeded())
                if _det.endswith('()'):
                    _name = _det[:-2]
                    return bool(getattr(_status, _name)())
            except Exception:
                return True
            return True

        def _p12v2_status_dump(_status):
            # Compact dump for a single status: Code, Message, Failed, etc.
            _info = {}
            for _name in ('Code', 'Message', 'Failed', 'Succeeded',
                          'InProgress', 'Init'):
                try:
                    _fn = getattr(_status, _name, None)
                    if callable(_fn):
                        _info[_name] = repr(_fn())[:120]
                except Exception:
                    pass
            return _info

        # Pre-compute uid range / wr total once, reuse for both BW summary
        # log and FAIL dump.
        try:
            _uid_lo = int(min(agg_xuids)) if agg_xuids else -1
            _uid_hi = int(max(agg_xuids)) if agg_xuids else -1
        except Exception:
            _uid_lo = _uid_hi = -1
        try:
            _total_wrs = sum(len(s) for s in agg_sizes)
        except Exception:
            _total_wrs = -1

        if _diag_bw and _batch_statuses:
            try:
                _bw_n_ok = sum(1 for _s in _batch_statuses if _p12v2_is_ok(_s))
                _bw_n_bad = len(_batch_statuses) - _bw_n_ok
                logger.warning(
                    "[PATCH-12-v2 DIAG-V4 BW] call#%d n_descs=%d total_wrs=%d "
                    "uid_range=[%d,%d] uid_span=%d ok=%d bad=%d",
                    _call_idx, len(agg_src_descs), _total_wrs,
                    _uid_lo, _uid_hi, _uid_hi - _uid_lo,
                    _bw_n_ok, _bw_n_bad,
                )
            except Exception as _e:
                logger.warning("[PATCH-12-v2 DIAG-V4 BW] log error: %s", _e)

        if _diag_failed and _batch_statuses:
            _bad_descs = 0
            _first_bad = None
            _bad_seg_idxs: List[int] = []
            _by_code_msg: Dict[str, int] = {}
            _python_check_pass_count = 0  # bad descs whose Python mirror passed
            for _di, _st in enumerate(_batch_statuses):
                if _p12v2_is_ok(_st):
                    continue
                _bad_descs += 1
                if _di < len(agg_seg_idx):
                    _bad_seg_idxs.append(int(agg_seg_idx[_di]))
                try:
                    _c = int(_st.Code())
                    _m = str(_st.Message())[:80]
                    _key = "code=%d msg=%r" % (_c, _m)
                    _by_code_msg[_key] = _by_code_msg.get(_key, 0) + 1
                except Exception:
                    pass
                # Python-side mirror of C++ bounds check (per-desc).
                if _di < len(agg_src_descs):
                    try:
                        _src_size_pc = int(getattr(agg_src_descs[_di], "size", 0) or 0)
                        _dst_size_pc = int(getattr(agg_dst_descs[_di], "size", 0) or 0)
                        _lops_pc = max(
                            (lo + s for lo, s in zip(agg_local_offsets[_di], agg_sizes[_di])),
                            default=0,
                        )
                        _rops_pc = max(
                            (ro + s for ro, s in zip(agg_remote_offsets[_di], agg_sizes[_di])),
                            default=0,
                        )
                        if _lops_pc <= _src_size_pc and _rops_pc <= _dst_size_pc:
                            _python_check_pass_count += 1
                    except Exception:
                        pass
                if _first_bad is None and _di < len(agg_src_descs):
                    _seg = agg_seg_idx[_di]
                    _il = int(item_lens[_seg])
                    # Python-side mirror of C++ bounds check at
                    # common.cpp:334: per-WR (off+sz) must be <= MR.length.
                    # We mirror it using desc.size, which is the size that
                    # Python registered via RegisterMemory. If THIS check
                    # PASSES while C++ returns 'length out of range', the
                    # C++ side's cached RdmaMemoryRegion.length disagrees
                    # with desc.size -- a smoking-gun signature of the
                    # 'remote.length=0 cached' race that MoRI's release
                    # build silently swallows (assert at backend_impl.cpp:
                    # 1129 is elided under -O3 -DNDEBUG).
                    _src_size = int(getattr(agg_src_descs[_di], "size", 0) or 0)
                    _dst_size = int(getattr(agg_dst_descs[_di], "size", 0) or 0)
                    _local_off_plus_sz_max = max(
                        (lo + s for lo, s in zip(agg_local_offsets[_di], agg_sizes[_di])),
                        default=0,
                    )
                    _remote_off_plus_sz_max = max(
                        (ro + s for ro, s in zip(agg_remote_offsets[_di], agg_sizes[_di])),
                        default=0,
                    )
                    _python_check_pass = (
                        _local_off_plus_sz_max <= _src_size
                        and _remote_off_plus_sz_max <= _dst_size
                    )
                    _first_bad = {
                        "call": _call_idx,
                        "desc_idx": _di,
                        "seg_idx": _seg,
                        "item_len": _il,
                        "uid": int(agg_xuids[_di]),
                        "src_id": int(getattr(agg_src_descs[_di], "id", -1) or -1),
                        "dst_id": int(getattr(agg_dst_descs[_di], "id", -1) or -1),
                        "src_size": _src_size,
                        "dst_size": _dst_size,
                        "local_off_plus_sz_max": _local_off_plus_sz_max,
                        "remote_off_plus_sz_max": _remote_off_plus_sz_max,
                        "python_check_pass": _python_check_pass,
                        "local_offs_head": list(agg_local_offsets[_di][:4]),
                        "remote_offs_head": list(agg_remote_offsets[_di][:4]),
                        "sizes_head": list(agg_sizes[_di][:4]),
                        "local_off_max": max(agg_local_offsets[_di]) if agg_local_offsets[_di] else -1,
                        "remote_off_max": max(agg_remote_offsets[_di]) if agg_remote_offsets[_di] else -1,
                        "size_max": max(agg_sizes[_di]) if agg_sizes[_di] else -1,
                        "n_wrs": len(agg_sizes[_di]),
                        "status_api": _p12v2_status_dump(_st),
                    }
            if _bad_descs > 0:
                logger.warning(
                    "[PATCH-12-v2 DIAG-V4 FAIL] call#%d bad_descs=%d/%d "
                    "py_pass=%d uid_range=[%d,%d] by_code_msg=%s "
                    "bad_seg_idxs=%s first_bad=%s",
                    _call_idx, _bad_descs, len(_batch_statuses),
                    _python_check_pass_count, _uid_lo, _uid_hi,
                    _by_code_msg, _bad_seg_idxs[:30], _first_bad,
                )
        return statuses

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[npt.NDArray[np.int32]] = None,
    ) -> Tuple[List[TransferStatus], Optional[List[TransferInfo]]]:
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        transfer_infos = self.transfer_infos.get(bootstrap_room)
        if not transfer_infos:
            raise RuntimeError(
                f"No transfer info found for bootstrap_room={bootstrap_room}"
            )
        result_statuses = []
        target_infos_snapshot: Optional[List[TransferInfo]] = None
        with self.transfer_lock:
            self.update_status(bootstrap_room, KVPoll.Transferring)
            for info in transfer_infos.values():
                peer_info = self.decode_kv_args_table.get(info.engine_key)
                if not peer_info:
                    self.record_failure(
                        bootstrap_room,
                        f"Peer info missing for engine {info.engine_key}",
                    )
                    raise RuntimeError(
                        f"Missing decode peer info for {info.engine_key}"
                    )
                if not info.is_dummy:
                    dst_indices_chunk = info.dst_kv_indices[index_slice]
                    statuses = self.send_kvcache(
                        peer_info, kv_indices, dst_indices_chunk
                    )
                    result_statuses.extend(statuses)
                    # [PATCH-12-v2] dsv4: transfer compress / indexer state
                    # segments using SWA-page indices that prefill.py (patch 13)
                    # computed and that the decode side handed us via the
                    # state_bytes wire slot.
                    dst_state_chunk = getattr(info, "dst_state_indices", None)
                    if (
                        state_indices is not None
                        and len(state_indices) > 0
                        and dst_state_chunk is not None
                        and dst_state_chunk.size > 0
                    ):
                        state_statuses = self._transfer_state_buffers(
                            peer_info, state_indices, dst_state_chunk
                        )
                        result_statuses.extend(state_statuses)
                if (
                    is_last
                    and aux_index is not None
                    and info.dst_aux_index >= 0
                    and self.pp_group.is_last_rank
                ):
                    result_statuses.extend(
                        self.send_aux(
                            peer_info, aux_index, info.dst_aux_index, bootstrap_room
                        )
                    )
            if is_last:
                self.update_status(bootstrap_room, KVPoll.Success)
                target_infos_snapshot = list(transfer_infos.values())
                self.transfer_infos.pop(bootstrap_room, None)
        return result_statuses, target_infos_snapshot


class MoriKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.transfer_statuses: List[TransferStatus] = []
        self.pending_infos: Optional[List[TransferInfo]] = None
        self.sent_last_chunk = False
        self.conclude_state: Optional[KVPoll] = None
        self.status_notified = False
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        # Special handling for cp
        if self.kv_mgr.enable_all_cp_ranks_for_transfer:
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            if not is_last:
                return
            else:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return
        statuses, infos = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            aux_index=self.aux_index if is_last else None,
            state_indices=state_indices,  # [PATCH-12-v2]
        )
        self.transfer_statuses.extend(statuses)
        if infos is not None:
            self.pending_infos = infos
            self.sent_last_chunk = True

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status == KVPoll.Bootstrapping:
            elapsed = time.time() - self.init_time
            if elapsed >= self.kv_mgr.bootstrap_timeout:
                reason = (
                    f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s "
                    "waiting for decode handshake"
                )
                self.kv_mgr.record_failure(self.bootstrap_room, reason)
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self._finalize_failure(reason)
                return KVPoll.Failed
            return status

        if status == KVPoll.Failed:
            self._finalize_failure()
            return KVPoll.Failed

        transfers_done = self._all_transfers_finished()
        if transfers_done:
            if self._has_transfer_error():
                reason = self._collect_failure_reason()
                self.kv_mgr.record_failure(self.bootstrap_room, reason)
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self._finalize_failure(reason)
                return KVPoll.Failed
            self._notify_decode(KVPoll.Success)
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        return KVPoll.Transferring if status == KVPoll.Success else status

    def _all_transfers_finished(self) -> bool:
        if not self.sent_last_chunk:
            return False
        if not self.transfer_statuses:
            return True
        return all(not status.InProgress() for status in self.transfer_statuses)

    def _has_transfer_error(self) -> bool:
        return any(status.Failed() for status in self.transfer_statuses)

    def _collect_failure_reason(self) -> str:
        for status in self.transfer_statuses:
            if status.Failed():
                return f"KV transfer failed: {status.Message()}"
        return "KV transfer failed due to unknown reason"

    def _notify_decode(
        self, status: KVPoll, failure_reason: Optional[str] = None
    ) -> None:
        if self.status_notified:
            return
        if self.pending_infos:
            self.kv_mgr.notify_decode_status(
                self.pending_infos, self.bootstrap_room, status, failure_reason
            )
        self.status_notified = True

    def _finalize_failure(self, failure_reason: Optional[str] = None) -> None:
        if self.conclude_state == KVPoll.Failed:
            return
        if failure_reason is None:
            failure_reason = self.kv_mgr.failure_records.get(
                self.bootstrap_room, "KV transfer failed"
            )
        self._notify_decode(KVPoll.Failed, failure_reason)
        self.conclude_state = KVPoll.Failed

    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)

    def failure_exception(self):
        if self.conclude_state is None:
            self._finalize_failure()
        self.clear()
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "KV transfer failed"
            )
        raise RuntimeError(failure_reason)

    def abort(self):
        super().abort()
        self._notify_decode(KVPoll.Failed, "Aborted by AbortReq.")


class MoriKVReceiver(CommonKVReceiver):

    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.init_time: Optional[float] = None

    def init(
        self,
        prefill_dp_rank: int,
    ):
        super().init(prefill_dp_rank)
        if self.bootstrap_room is None:
            return
        self.kv_mgr.room_to_bootstrap_addr[self.bootstrap_room] = self.bootstrap_addr

    def _register_kv_args(self):
        if self.bootstrap_infos is None:
            return
        engine_desc_blob = self.kv_mgr.engine_desc.pack()
        packed_kv_descs = _pack_mem_desc_list(self.kv_mgr.kv_mem_descs)
        packed_aux_descs = _pack_mem_desc_list(self.kv_mgr.aux_mem_descs)
        packed_state_descs = _pack_mem_desc_list(self.kv_mgr.state_mem_descs)
        gpu_id = str(self.kv_mgr.kv_args.gpu_id).encode("ascii")
        decode_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
        decode_tp_rank = str(self.kv_mgr.kv_args.engine_rank).encode("ascii")
        kv_item_len = str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii")

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        MORI_GUARD,
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        engine_desc_blob,
                        packed_kv_descs,
                        packed_aux_descs,
                        packed_state_descs,
                        gpu_id,
                        decode_tp_size,
                        decode_tp_rank,
                        kv_item_len,
                    ]
                )

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        if self.bootstrap_infos is None or self.bootstrap_room is None:
            return

        kv_indices_bytes = (
            np.asarray(kv_indices, dtype=np.int32).tobytes() if kv_indices.size else b""
        )
        aux_bytes = str(aux_index).encode("ascii") if aux_index is not None else b""
        # [PATCH-12-v2] dsv4: populate the state-indices wire slot.
        if state_indices is not None and len(state_indices) > 0:
            state_bytes = np.asarray(state_indices, dtype=np.int32).tobytes()
        else:
            state_bytes = b""

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info.get("is_dummy", False)
            with lock:
                sock.send_multipart(
                    [
                        MORI_GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.engine_desc.key.encode("ascii"),
                        kv_indices_bytes if not is_dummy else b"",
                        aux_bytes if not is_dummy else b"",
                        state_bytes,
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status

        if status == KVPoll.WaitingForInput and self.init_time is not None:
            elapsed = time.time() - self.init_time
            if elapsed >= self.kv_mgr.waiting_timeout:
                reason = f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s waiting for KV transfer"
                self.kv_mgr.record_failure(self.bootstrap_room, reason)
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self.conclude_state = KVPoll.Failed
                return KVPoll.Failed

        return status

    def clear(self) -> None:
        if self.bootstrap_room is None:
            return
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room, None)
        self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room, None)
        self.kv_mgr._cleanup_room_tracking(self.bootstrap_room)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "KV transfer failed"
            )
        raise RuntimeError(failure_reason)

    def abort(self):
        if self.bootstrap_room is None:
            return
        super().abort()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.clear()


class MoriKVBootstrapServer(CommonKVBootstrapServer):
    pass
