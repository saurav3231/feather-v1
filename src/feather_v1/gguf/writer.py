"""Feather v1 -- GGUF v3 writer (production, little-endian, aligned 32).

Header is exactly 24 bytes::

    magic "GGUF" (4) + version uint32 (4) + tensor_count uint64 (8)
    + metadata_kv_count uint64 (8)

then metadata KVs, tensor infos, padded data aligned to 32 bytes.

Supported tensor types:

===========  =====  =========================================================
name         id    layout (verified self round-trip)
===========  =====  =========================================================
F32          0      raw float32
F16          1      raw float16 (llama.cpp standard)
Q8_0         8      block of 32: f16 d + int8 q[32]; y = q*d (llama.cpp std)
Q4_K         12     block of 256: f16 d + f16 dmin + uint8 scales[12]
                    + uint8 qs[64] (2-bit index, 4 elems/byte); y =
                    dmin + d*sc_g*idx.  feather-documented layout, verified
                    by :mod:`feather_v1.gguf.reader`; llama.cpp validation
                    is pending (see docs/DISTRIBUTION.md).
===========  =====  =========================================================

``Q4_K_M`` in GGUF is a *file type* (general.file_type = 15) composed of
Q4_K tensors -- it is not a tensor type.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGN = 32

# metadata value types (gguf.md v3)
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32 = 0, 1, 2, 3, 4, 5, 6
BOOL, STRING = 7, 8
ARRAY = 9
UINT64, INT64, FLOAT64 = 10, 11, 12

# tensor types (ggml)
F32, F16, Q8_0, Q4_K = 0, 1, 8, 12

# general.file_type values (ggml model types)
FILE_F16 = 1
FILE_MOSTLY_Q8_0 = 7
FILE_MOSTLY_Q4_K_M = 15
FILE_F32 = 0


@dataclass
class PackedTensor:
    """One tensor's GGUF representation."""

    name: str
    ggml_type: int
    dims: tuple[int, ...]
    payload: bytes
    n_elements: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_elements = int(np.prod(self.dims, dtype=np.int64))


def _write_string(fh, value: str) -> None:
    raw = value.encode("utf-8")
    fh.write(struct.pack("<Q", len(raw)))
    fh.write(raw)


def _write_value(fh, vtype: int, value: Any) -> None:
    if vtype == UINT8:
        fh.write(struct.pack("<B", int(value)))
    elif vtype == INT8:
        fh.write(struct.pack("<b", int(value)))
    elif vtype == BOOL:
        fh.write(struct.pack("<?", bool(value)))
    elif vtype == UINT16:
        fh.write(struct.pack("<H", int(value)))
    elif vtype == INT16:
        fh.write(struct.pack("<h", int(value)))
    elif vtype == UINT32:
        fh.write(struct.pack("<I", int(value)))
    elif vtype == INT32:
        fh.write(struct.pack("<i", int(value)))
    elif vtype == FLOAT32:
        fh.write(struct.pack("<f", float(value)))
    elif vtype == UINT64:
        fh.write(struct.pack("<Q", int(value)))
    elif vtype == INT64:
        fh.write(struct.pack("<q", int(value)))
    elif vtype == FLOAT64:
        fh.write(struct.pack("<d", float(value)))
    elif vtype == STRING:
        _write_string(fh, str(value))
    elif vtype == ARRAY:
        elem_type, items = value
        fh.write(struct.pack("<I", int(elem_type)))
        fh.write(struct.pack("<Q", len(items)))
        for item in items:
            _write_value(fh, elem_type, item)
    else:
        raise ValueError(f"unsupported gguf value type {vtype}")


def _write_tensor_info(fh, tensor: PackedTensor, offset: int) -> None:
    _write_string(fh, tensor.name)
    fh.write(struct.pack("<I", len(tensor.dims)))
    for dim in tensor.dims[::-1]:
        fh.write(struct.pack("<Q", dim))
    fh.write(struct.pack("<I", tensor.ggml_type))
    fh.write(struct.pack("<Q", offset))


# -- quantization -----------------------------------------------------------
def f32_pack(name: str, arr: np.ndarray) -> PackedTensor:
    return PackedTensor(
        name, F32, arr.shape, np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    )


def f16_pack(name: str, arr: np.ndarray) -> PackedTensor:
    return PackedTensor(
        name, F16, arr.shape, np.ascontiguousarray(arr, dtype=np.float16).tobytes()
    )


def q8_0_pack(name: str, arr: np.ndarray) -> PackedTensor:
    """llama.cpp Q8_0: blocks of 32 -> f16 d + int8 q[32]; y = q*d."""
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    n = flat.size
    pad = (-n) % 32
    work = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = work.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    d = np.maximum(amax, 1e-12) / 127.0
    q = np.clip(np.round(blocks / d[:, None]), -127.0, 127.0).astype(np.int8)
    out = bytearray()
    for i in range(blocks.shape[0]):
        out += d[i].astype(np.float16).tobytes()
        out += q[i].tobytes()
    return PackedTensor(name, Q8_0, arr.shape, bytes(out))


def q4_k_pack(name: str, arr: np.ndarray) -> PackedTensor:
    """Q4_K-style 2-bit packing, feather-documented layout.

    Per 256-elem block: ``f16 d`` (global step), ``f16 dmin`` (block min),
    ``uint8 scales[12]`` (one nibble per 32-elem group, low nibble used),
    ``uint8 qs[64]`` (3 2-bit indices packed per byte, 4 elems/byte).
    Dequant (reader): ``y = dmin + d * sc_g * idx`` - exact inverse.
    """
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    n = flat.size
    pad = (-n) % 256
    work = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = work.reshape(-1, 256)
    out = bytearray()
    eps_sc = 1e-9
    for blk in blocks:
        b_min = float(blk.min())
        b_max = float(blk.max())
        d = (b_max - b_min) / 45.0
        if d <= 0:
            d = eps_sc
        groups = blk.reshape(8, 32)
        scales = []
        indices: list[bytes] = []
        for g in groups:
            g_min = float(g.min())
            g_max = float(g.max())
            spread = g_max - g_min
            sc_g = int(max(1, min(15, np.ceil(spread / (3.0 * d)))))
            scaled = (g - b_min) / (d * sc_g)
            idx = np.clip(np.round(scaled), 0, 3).astype(np.uint8)
            scales.append(sc_g)
            packed = np.zeros(8, dtype=np.uint8)
            for k in range(32):
                packed[k // 4] |= idx[k] << (2 * (k % 4))
            indices.append(packed.tobytes())
        out += np.float16(d).tobytes()
        out += np.float16(b_min).tobytes()
        out += bytes(scales) + b"\x00\x00\x00\x00"
        out += b"".join(indices)
    return PackedTensor(name, Q4_K, arr.shape, bytes(out))


QUANTIZERS = {
    "f32": f32_pack,
    "f16": f16_pack,
    "q8_0": q8_0_pack,
    "q8": q8_0_pack,
    "q4_k_m": q4_k_pack,
    "q4_k": q4_k_pack,
}

FILE_TYPES = {
    "f32": FILE_F32,
    "f16": FILE_F16,
    "q8_0": FILE_MOSTLY_Q8_0,
    "q8": FILE_MOSTLY_Q8_0,
    "q4_k_m": FILE_MOSTLY_Q4_K_M,
    "q4_k": FILE_MOSTLY_Q4_K_M,
}


def _align_up(pos: int, boundary: int = ALIGN) -> int:
    return (pos + boundary - 1) // boundary * boundary


def write_gguf(
    path: str | Path,
    tensors: list[PackedTensor],
    metadata: list[tuple[str, int, Any]],
) -> Path:
    """Write spec-correct GGUF v3 (24-byte header, 32-aligned data).

    Tensor offsets stored in the file are relative to the aligned start of
    the data section (the gguf.md convention; llama.cpp adds
    ``gguf_get_data_offset``).  The last payload is not padded.
    """
    path = Path(path)
    offsets: list[int] = []
    off = 0
    for t in tensors:
        off = _align_up(off)
        offsets.append(off)
        off += len(t.payload)

    head = io.BytesIO()
    head.write(GGUF_MAGIC)
    head.write(struct.pack("<I", GGUF_VERSION))
    head.write(struct.pack("<Q", len(tensors)))
    head.write(struct.pack("<Q", len(metadata)))
    for key, vtype, value in metadata:
        _write_string(head, key)
        head.write(struct.pack("<I", vtype))
        _write_value(head, vtype, value)
    for t, offset in zip(tensors, offsets):
        _write_tensor_info(head, t, offset)
    data_start = _align_up(head.tell())
    head_bytes = head.getvalue() + b"\x00" * (data_start - head.tell())

    with open(path, "wb") as fh:
        fh.write(head_bytes)
        for t, offset in zip(tensors, offsets):
            pos = fh.tell()
            if pos % ALIGN:
                fh.seek(_align_up(pos))
            target = data_start + offset
            assert (
                fh.tell() == target
            ), f"offset mismatch {fh.tell()} != {target} at {t.name}"
            fh.write(t.payload)
    return path
