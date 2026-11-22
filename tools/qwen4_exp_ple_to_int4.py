"""Convert the Flash Next PLE n-gram table from FP8 (one global scale) to INT4.

Each 160-value row becomes 90 bytes: 80 bytes of 4-bit codes (value j in the
low nibble of byte j//2 for even j, high nibble for odd j; code = q + 8 with
q in [-7, 7]), then 5 fp16 scales, one per group of 32 values, stored times
SCALE_MULT (keeps small scales out of the fp16 subnormal range). A value is
(code - 8) * scale / SCALE_MULT; the global FP8 scale is folded in.

python3 ple_to_int4.py <fp8 table dir> <out dir>
"""
import glob
import json
import os
import struct
import sys
import time

import numpy as np

GROUP = 32
SCALE_MULT = 1024.0
CHUNK = 1 << 20  # rows per step


def fp8_e4m3fn_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 7
        if e == 15 and m == 7:
            lut[b] = 0.0  # NaN in e4m3fn; never a table value
        elif e == 0:
            lut[b] = s * (m / 8.0) * 2.0**-6
        else:
            lut[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return lut


def read_scalar(raw: bytes, dtype: str) -> float:
    if dtype == "F32":
        return float(np.frombuffer(raw, dtype=np.float32)[0])
    if dtype == "F16":
        return float(np.frombuffer(raw, dtype=np.float16)[0])
    if dtype == "BF16":  # top half of a float32
        u = np.frombuffer(raw, dtype=np.uint16)[:1].astype(np.uint32) << 16
        return float(u.view(np.float32)[0])
    raise ValueError(f"unsupported scale dtype {dtype}")


def convert_rows(raw: np.ndarray, lut: np.ndarray, gscale: float) -> np.ndarray:
    n, d = raw.shape
    v = lut[raw] * np.float32(gscale)
    g = v.reshape(n, d // GROUP, GROUP)
    s = np.abs(g).max(axis=-1) / np.float32(7.0)
    s16 = (s * np.float32(SCALE_MULT)).astype(np.float16)
    s_eff = s16.astype(np.float32) / np.float32(SCALE_MULT)
    safe = np.where(s_eff > 0, s_eff, np.float32(1.0))
    q = np.clip(np.rint(g / safe[..., None]), -7, 7).astype(np.int8)
    q = np.where(s_eff[..., None] > 0, q, 0).reshape(n, d)
    codes = (q + 8).astype(np.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    out = np.empty((n, d // 2 + 2 * (d // GROUP)), dtype=np.uint8)
    out[:, : d // 2] = packed
    out[:, d // 2 :] = s16.view(np.uint8).reshape(n, -1)
    return out


def main(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    lut = fp8_e4m3fn_lut()
    paths = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    headers = {}
    gscale = None
    for p in paths:
        with open(p, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(hl))
        headers[p] = (hl, h)
        for name, info in h.items():
            if name.endswith(".ngram_embedding.weight_scale"):
                with open(p, "rb") as f:
                    f.seek(8 + hl + info["data_offsets"][0])
                    raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
                gscale = read_scalar(raw, info["dtype"])
    assert gscale is not None, "no weight_scale"
    print(f"global scale {gscale:.6g}, {len(paths)} files", flush=True)
    t0 = time.time()
    total = 0
    for p in paths:
        hl, h = headers[p]
        shards = {k: v for k, v in h.items() if ".ngram_embedding.shard_" in k}
        if not shards:
            continue
        out_path = os.path.join(dst, os.path.basename(p))
        if os.path.exists(out_path):
            print("skip existing", out_path, flush=True)
            continue
        new_h = {"__metadata__": {"format": "pt", "ple_quant": "int4g32",
                                  "scale_mult": str(SCALE_MULT)}}
        off = 0
        order = sorted(shards)
        for k in order:
            rows, d = shards[k]["shape"]
            nb = rows * (d // 2 + 2 * (d // GROUP))
            new_h[k] = {"dtype": "U8", "shape": [rows, d // 2 + 2 * (d // GROUP)],
                        "data_offsets": [off, off + nb]}
            off += nb
        hb = json.dumps(new_h, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        tmp = out_path + ".tmp"
        with open(p, "rb") as fi, open(tmp, "wb") as fo:
            fo.write(struct.pack("<Q", len(hb)))
            fo.write(hb)
            for k in order:
                info = shards[k]
                assert info["dtype"] == "F8_E4M3", info
                rows, d = info["shape"]
                base = 8 + hl + info["data_offsets"][0]
                for r0 in range(0, rows, CHUNK):
                    r1 = min(rows, r0 + CHUNK)
                    fi.seek(base + r0 * d)
                    raw = np.frombuffer(fi.read((r1 - r0) * d), dtype=np.uint8).reshape(-1, d)
                    fo.write(convert_rows(raw, lut, gscale).tobytes())
                total += rows
        os.replace(tmp, out_path)
        print(f"{os.path.basename(p)}: {total} rows so far, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {total} rows in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
