# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-shot peer-to-peer all-reduce for Intel GPUs over PCIe.

oneCCL costs about 21 us per all-reduce on 3x Arc Pro B70, however small the
message. Decode all-reduces are only tokens x hidden_size, so that fixed cost
dominates. This module pushes each rank's data straight into the other GPUs'
memory and sums it there (about 10 us for 1-6 tokens at TP=3). Large messages
stay on oneCCL. The kernels and the reasons behind their design are in
csrc/xpu/p2p_all_reduce.cpp and docs/design/xpu_p2p_all_reduce.md.

The kernels live in a small SYCL library loaded with ctypes. Build it with
tools/build_xpu_p2p_all_reduce.sh.
"""

import ctypes
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_LIB_NAME = "_xpu_p2p_all_reduce.so"
_DTYPE_CODES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_IPC_HANDLE_SIZE = 64
_ERR_LEN = 512

DEFAULT_MAX_BYTES = 140 * 1024  # ~14 tokens of a 5120-wide fp16 model


def _load_library() -> ctypes.CDLL:
    path = os.environ.get("VLLM_XPU_P2P_ALLREDUCE_LIB") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), _LIB_NAME
    )
    lib = ctypes.CDLL(path)
    vp, i, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    cp = ctypes.c_char_p
    lib.xpu_p2p_ar_create.argtypes = [vp, i, i, u64, cp, i]
    lib.xpu_p2p_ar_create.restype = vp
    lib.xpu_p2p_ar_get_ipc_handle.argtypes = [vp, cp]
    lib.xpu_p2p_ar_get_ipc_handle.restype = i
    lib.xpu_p2p_ar_open_peer.argtypes = [vp, i, i, cp, cp, i]
    lib.xpu_p2p_ar_open_peer.restype = i
    lib.xpu_p2p_ar_warmup.argtypes = [vp, vp, cp, i]
    lib.xpu_p2p_ar_warmup.restype = i
    lib.xpu_p2p_ar_all_reduce.argtypes = [vp, vp, vp, vp, u64, i]
    lib.xpu_p2p_ar_all_reduce.restype = i
    f32 = ctypes.c_float
    lib.xpu_p2p_ar_fused_add_rms_norm.argtypes = [
        *(vp, vp, vp, vp, vp, vp, vp),
        *(u64, u64, i, i, f32, f32),
    ]
    lib.xpu_p2p_ar_fused_add_rms_norm.restype = i
    lib.xpu_p2p_ar_timeouts.argtypes = [vp, vp]
    lib.xpu_p2p_ar_timeouts.restype = u64
    lib.xpu_p2p_ar_destroy.argtypes = [vp]
    lib.xpu_p2p_ar_destroy.restype = None
    return lib


class XpuP2PAllReduce:
    """All-reduce for small tensors over direct GPU-to-GPU PCIe writes.

    Set-up is collective over ``cpu_group``. If it fails on any rank, it is
    disabled on every rank, so all ranks always take the same path (a rank on
    oneCCL and another on this path would hang).
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ):
        self.disabled = True
        self._state = None
        self._lib = None
        self.group = cpu_group
        self.rank = dist.get_rank(group=cpu_group)
        self.world_size = dist.get_world_size(group=cpu_group)
        self.device = device
        self.max_bytes = max_bytes - max_bytes % 4

        ok, why = True, ""
        handle = bytes(_IPC_HANDLE_SIZE)
        try:
            if self.world_size < 2 or self.world_size > 8:
                raise RuntimeError(f"world size {self.world_size} not in 2..8")
            self._lib = _load_library()
            err = ctypes.create_string_buffer(_ERR_LEN)
            self._state = self._lib.xpu_p2p_ar_create(
                self._queue(),
                self.rank,
                self.world_size,
                self.max_bytes,
                err,
                _ERR_LEN,
            )
            if not self._state:
                raise RuntimeError(err.value.decode())
            buf = ctypes.create_string_buffer(_IPC_HANDLE_SIZE)
            rc = self._lib.xpu_p2p_ar_get_ipc_handle(self._state, buf)
            if rc != 0:
                raise RuntimeError(f"zeMemGetIpcHandle failed (0x{rc:x})")
            handle = buf.raw
        except Exception as e:  # noqa: BLE001 - any failure disables the path
            ok, why = False, f"set-up: {e}"

        # Step 1: everyone shares (ok, pid, handle); only continue if all are ok.
        infos: list = [None] * self.world_size
        dist.all_gather_object(infos, (ok, os.getpid(), handle), group=cpu_group)
        if all(info[0] for info in infos):
            try:
                err = ctypes.create_string_buffer(_ERR_LEN)
                for peer, (_, pid, peer_handle) in enumerate(infos):
                    if peer == self.rank:
                        continue
                    rc = self._lib.xpu_p2p_ar_open_peer(
                        self._state, peer, pid, peer_handle, err, _ERR_LEN
                    )
                    if rc != 0:
                        raise RuntimeError(err.value.decode())
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"mapping peer buffers: {e}"
        else:
            ok = False
            why = why or "another rank failed set-up"

        # Step 2: agree again, then warm up (touch peer memory) between barriers.
        oks: list = [None] * self.world_size
        dist.all_gather_object(oks, ok, group=cpu_group)
        if all(oks):
            dist.barrier(group=cpu_group)
            err = ctypes.create_string_buffer(_ERR_LEN)
            rc = self._lib.xpu_p2p_ar_warmup(self._state, self._queue(), err, _ERR_LEN)
            if rc != 0:
                ok, why = False, f"warm-up: {err.value.decode()}"
            oks = [None] * self.world_size
            dist.all_gather_object(oks, ok, group=cpu_group)

        if all(oks):
            self.disabled = False
            logger.info(
                "XPU P2P all-reduce enabled: rank %d/%d, messages up to %d bytes",
                self.rank,
                self.world_size,
                self.max_bytes,
            )
        else:
            logger.warning(
                "XPU P2P all-reduce disabled, using oneCCL only (%s)",
                why or "another rank failed",
            )
            self.close()

    def _queue(self) -> int:
        return torch.xpu.current_stream(self.device).sycl_queue

    def should_use(self, inp: torch.Tensor, limit: int | None = None) -> bool:
        """Whether ``inp`` can take this path. ``limit`` (at most max_bytes)
        lowers the size limit for this call."""
        if self.disabled or inp.dtype not in _DTYPE_CODES:
            return False
        nbytes = inp.numel() * inp.element_size()
        limit = self.max_bytes if limit is None else min(limit, self.max_bytes)
        # Same answer on every rank: it depends only on shape and dtype.
        return 0 < nbytes <= limit and nbytes % 4 == 0 and inp.is_contiguous()

    def all_reduce(self, inp: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        """Sum ``inp`` over all ranks. Returns ``inp`` itself if ``inplace``."""
        if inp.data_ptr() % 4:
            # The kernels move 4-byte words. Rare (odd offset views); copy.
            inp, inplace = inp.clone(), False
        out = inp if inplace else torch.empty_like(inp)
        rc = self._lib.xpu_p2p_ar_all_reduce(
            self._state,
            self._queue(),
            inp.data_ptr(),
            out.data_ptr(),
            inp.numel() * inp.element_size(),
            _DTYPE_CODES[inp.dtype],
        )
        if rc != 0:
            raise RuntimeError(f"xpu_p2p_ar_all_reduce failed ({rc})")
        return out

    def fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        weight_bias: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """All-reduce ``inp`` [..., hidden], then do vllm_ir fused_add_rms_norm
        with ``residual`` and ``weight + weight_bias`` in the same kernel.
        Returns (normed, new_residual), or None when this call can't take the
        P2P path (the caller must then use the unfused ops). The answer
        depends only on shapes and dtypes, so it is the same on all ranks."""
        hidden = inp.shape[-1]
        if (
            not self.should_use(inp)
            or residual.dtype != inp.dtype
            or residual.shape != inp.shape
            or not residual.is_contiguous()
            or weight.dtype not in _DTYPE_CODES
            or weight.numel() != hidden
            or not weight.is_contiguous()
            or inp.data_ptr() % 4
        ):
            return None
        rows = inp.numel() // hidden
        out = torch.empty_like(inp)
        res_out = torch.empty_like(inp)
        rc = self._lib.xpu_p2p_ar_fused_add_rms_norm(
            self._state,
            self._queue(),
            inp.data_ptr(),
            residual.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            res_out.data_ptr(),
            rows,
            hidden,
            _DTYPE_CODES[inp.dtype],
            _DTYPE_CODES[weight.dtype],
            float(weight_bias),
            float(epsilon),
        )
        if rc == -1:
            return None
        if rc != 0:
            raise RuntimeError(f"xpu_p2p_ar_fused_add_rms_norm failed ({rc})")
        return out, res_out

    def timeouts(self) -> int:
        """Polls that gave up waiting for a peer (should be 0). Synchronizes."""
        if self._state is None:
            return 0
        return int(self._lib.xpu_p2p_ar_timeouts(self._state, self._queue()))

    def close(self) -> None:
        self.disabled = True
        if self._state is not None and self._lib is not None:
            self._lib.xpu_p2p_ar_destroy(self._state)
        self._state = None
