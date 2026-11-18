# SPDX-License-Identifier: Apache-2.0
"""Minimal GGUF reader for the DeepSeek V4.1 XPU port.

It parses the header of every shard of a split GGUF (``*-0000i-of-0000n.gguf``),
records each tensor's type, shape and absolute byte range, and reads tensors
with ``os.pread`` (no whole-file mmap, so the 110 GB Engram shard is fine).
Rows of the Engram table are read with ``mmap`` slices (random access, one row
per hashed n-gram).

Shapes follow GGUF order (ne0 first, ne0 = innermost = input features). The
helpers here return torch shapes in row-major order (reversed).
"""
from __future__ import annotations

import glob
import mmap
import os
import re
import struct
from dataclasses import dataclass

import numpy as np
import torch

# ggml type id -> (name, block elements, block bytes)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}
TYPE_ID = {v[0]: k for k, v in GGML_TYPES.items()}
_PLAIN = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
           11: "q", 12: "d"}


@dataclass
class GTensor:
    name: str
    type_id: int
    dims: tuple[int, ...]  # GGUF order, ne0 first
    path: str
    offset: int  # absolute file offset of the data
    nbytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES[self.type_id][0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Row-major torch shape (reversed GGUF dims)."""
        return tuple(reversed(self.dims))

    @property
    def row_bytes(self) -> int:
        """Bytes per row of ne0 elements."""
        _, be, bb = GGML_TYPES[self.type_id]
        return self.dims[0] // be * bb


class _Hdr:
    def __init__(self, f):
        self.f = f

    def rd(self, fmt):
        n = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.f.read(n))[0]

    def rstr(self):
        return self.f.read(self.rd("Q")).decode("utf-8", "replace")

    def rval(self, t):
        if t == 8:
            return self.rstr()
        if t == 9:
            et, n = self.rd("I"), self.rd("Q")
            if et in _SCALAR and et != 7:
                fmt = _SCALAR[et]
                sz = struct.calcsize("<" + fmt)
                return list(np.frombuffer(self.f.read(sz * n), dtype="<" + {
                    "B": "u1", "b": "i1", "H": "u2", "h": "i2", "I": "u4", "i": "i4",
                    "f": "f4", "Q": "u8", "q": "i8", "d": "f8"}[fmt]).tolist())
            return [self.rval(et) for _ in range(n)]
        return self.rd(_SCALAR[t])


def _parse(path: str, keep_meta: bool):
    meta: dict = {}
    tensors: list[GTensor] = []
    with open(path, "rb") as f:
        h = _Hdr(f)
        assert f.read(4) == b"GGUF", path
        ver, nt, nkv = h.rd("I"), h.rd("Q"), h.rd("Q")
        assert ver == 3, f"{path}: GGUF version {ver}"
        for _ in range(nkv):
            k = h.rstr()
            t = h.rd("I")
            if not keep_meta and k == "tokenizer.ggml.tokens":
                pass
            meta[k] = h.rval(t)
        infos = []
        for _ in range(nt):
            name = h.rstr()
            nd = h.rd("I")
            dims = tuple(h.rd("Q") for _ in range(nd))
            tid = h.rd("I")
            off = h.rd("Q")
            infos.append((name, dims, tid, off))
        align = int(meta.get("general.alignment", 32))
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
    for name, dims, tid, off in infos:
        _, be, bb = GGML_TYPES[tid]
        n = int(np.prod(dims))
        tensors.append(GTensor(name, tid, dims, path, data_start + off, n // be * bb))
    return meta, tensors


class GGUFCheckpoint:
    """All shards of a split GGUF. ``path`` is any shard or the folder."""

    _cache: dict[str, "GGUFCheckpoint"] = {}

    @classmethod
    def get(cls, path: str) -> "GGUFCheckpoint":
        key = os.path.realpath(path)
        if key not in cls._cache:
            cls._cache[key] = cls(path)
        return cls._cache[key]

    def __init__(self, path: str):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*-0000[0-9]-of-0000[0-9].gguf")))
            if not files:
                files = sorted(glob.glob(os.path.join(path, "*.gguf")))
        else:
            m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", path)
            if m:
                files = [f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf"
                         for i in range(1, int(m.group(2)) + 1)]
            else:
                files = [path]
        files = [f for f in files if "mmproj" not in os.path.basename(f)]
        self.files = files
        self.meta: dict = {}
        self.tensors: dict[str, GTensor] = {}
        for i, p in enumerate(files):
            meta, ts = _parse(p, keep_meta=(i == 0))
            for k, v in meta.items():
                self.meta.setdefault(k, v)
            for t in ts:
                self.tensors[t.name] = t
        self._fds: dict[str, int] = {}
        self._maps: dict[str, mmap.mmap] = {}

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def raw(self, name: str, start: int = 0, length: int | None = None) -> torch.Tensor:
        """Raw bytes [start, start+length) of a tensor as a uint8 CPU tensor."""
        t = self.tensors[name]
        if length is None:
            length = t.nbytes - start
        assert 0 <= start and start + length <= t.nbytes, (name, start, length)
        buf = bytearray(length)
        mv = memoryview(buf)
        fd = self._fd(t.path)
        done = 0
        while done < length:
            n = os.preadv(fd, [mv[done:]], t.offset + start + done)
            if n <= 0:
                raise OSError(f"short read of {name}")
            done += n
        return torch.frombuffer(buf, dtype=torch.uint8)

    def plain(self, name: str) -> torch.Tensor:
        """An F32/F16/BF16 tensor in torch shape."""
        t = self.tensors[name]
        dt = _PLAIN[t.type_name]
        return self.raw(name).view(dt).reshape(t.shape)

    def rows_map(self, name: str) -> memoryview:
        """A read-only mmap view of a tensor's bytes (for row lookups)."""
        t = self.tensors[name]
        mm = self._maps.get(t.path)
        if mm is None:
            fd = self._fd(t.path)
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
            try:
                mm.madvise(mmap.MADV_RANDOM)
            except (AttributeError, OSError):
                pass
            self._maps[t.path] = mm
        return memoryview(mm)[t.offset:t.offset + t.nbytes]


# ---------------------------------------------------------------- IQ4_NL rows
_IQ4NL_VALUES = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10,
                              1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.float32)


def dequant_iq4_nl(raw: torch.Tensor, n_values: int) -> torch.Tensor:
    """raw: uint8 [..., n_values/32*18] -> float32 [..., n_values].
    Block = fp16 scale d, then 16 bytes: low nibbles give values 0..15, high
    nibbles give values 16..31."""
    lead = raw.shape[:-1]
    b = raw.reshape(-1, 18)
    d = b[:, :2].contiguous().view(torch.float16).float()  # [nb, 1]
    q = b[:, 2:]
    lo = (q & 0x0F).long()
    hi = (q >> 4).long()
    vals = _IQ4NL_VALUES.to(raw.device)
    out = torch.cat([vals[lo], vals[hi]], dim=1) * d
    return out.reshape(*lead, n_values)
