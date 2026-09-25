"""Feather v1 -- GGUF v3 reader (little-endian).

Self-validating: reads header (24 bytes), metadata KVs (including ARRAY
values), tensor infos and re-dequantizes F32 / F16 / Q8_0 / Q4_K payloads
back to float32 -- the exact inverse of :mod:`feather_v1.gguf.writer`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .writer import F16, F32, GGUF_MAGIC, Q4_K, Q8_0

# metadata value types (mirrors writer)
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL = 0, 1, 2, 3, 4, 5, 6, 7
_STR, _ARR, _U64, _I64, _F64 = 8, 9, 10, 11, 12


@dataclass
class GGUFTensor:
    name: str
    ggml_type: int
    dims: tuple[int, ...]
    offset: int
    blob: bytes
    n_elements: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_elements = int(np.prod(self.dims, dtype=np.int64))

    def dequant(self) -> np.ndarray:
        if self.ggml_type == F32:
            return (
                np.frombuffer(self.blob, dtype="<f4")
                .astype(np.float32)
                .reshape(self.dims)
            )
        if self.ggml_type == F16:
            return (
                np.frombuffer(self.blob, dtype="<f2")
                .astype(np.float32)
                .reshape(self.dims)
            )
        if self.ggml_type == Q8_0:
            return _dequant_q8_0(self.blob, self.n_elements, self.dims)
        if self.ggml_type == Q4_K:
            return _dequant_q4_k(self.blob, self.n_elements, self.dims)
        raise ValueError(f"unsupported tensor type {self.ggml_type}")


def _dequant_q8_0(blob: bytes, n: int, dims: tuple[int, ...]) -> np.ndarray:
    blocks = len(blob) // 34  # f16 d (2) + int8 q (32)
    out = np.empty(blocks * 32, dtype=np.float32)
    for i in range(blocks):
        d = np.frombuffer(blob[i * 34 : i * 34 + 2], dtype="<f2").astype(np.float32)[0]
        q = np.frombuffer(blob[i * 34 + 2 : i * 34 + 34], dtype="<i1").astype(
            np.float32
        )
        out[i * 32 : (i + 1) * 32] = q * d
    return out[:n].reshape(dims)


def _dequant_q4_k(blob: bytes, n: int, dims: tuple[int, ...]) -> np.ndarray:
    block_bytes = 2 + 2 + 12 + 64  # 80
    nblocks = len(blob) // block_bytes
    out = np.empty(nblocks * 256, dtype=np.float32)
    for b in range(nblocks):
        seg = blob[b * block_bytes : (b + 1) * block_bytes]
        d = np.frombuffer(seg[0:2], dtype="<f2").astype(np.float32)[0]
        dmin = np.frombuffer(seg[2:4], dtype="<f2").astype(np.float32)[0]
        scales = np.frombuffer(seg[4:16], dtype=np.uint8)
        qs = np.frombuffer(seg[16:80], dtype=np.uint8)
        base = b * 256
        for g in range(8):
            sc = float(int(scales[g] & 0x0F))
            for k in range(32):
                byte_idx = g * 32 + k
                val = int((qs[byte_idx // 4] >> (2 * (byte_idx % 4))) & 0x03)
                out[base + g * 32 + k] = dmin + d * sc * val
    return out[:n].reshape(dims)


def _read_string(raw: bytes, pos: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", raw, pos)
    pos += 8
    value = raw[pos : pos + n].decode("utf-8")
    return value, pos + n


def _read_value(raw: bytes, pos: int, vtype: int) -> tuple[Any, int]:
    if vtype == _U8:
        return struct.unpack_from("<B", raw, pos)[0], pos + 1
    if vtype == _I8:
        return struct.unpack_from("<b", raw, pos)[0], pos + 1
    if vtype == _BOOL:
        return bool(struct.unpack_from("<B", raw, pos)[0]), pos + 1
    if vtype == _U16:
        return struct.unpack_from("<H", raw, pos)[0], pos + 2
    if vtype == _I16:
        return struct.unpack_from("<h", raw, pos)[0], pos + 2
    if vtype == _U32:
        return struct.unpack_from("<I", raw, pos)[0], pos + 4
    if vtype == _I32:
        return struct.unpack_from("<i", raw, pos)[0], pos + 4
    if vtype == _F32:
        return struct.unpack_from("<f", raw, pos)[0], pos + 4
    if vtype == _F64:
        return struct.unpack_from("<d", raw, pos)[0], pos + 8
    if vtype == _U64:
        return struct.unpack_from("<Q", raw, pos)[0], pos + 8
    if vtype == _I64:
        return struct.unpack_from("<q", raw, pos)[0], pos + 8
    if vtype == _STR:
        return _read_string(raw, pos)
    if vtype == _ARR:
        (elem,) = struct.unpack_from("<I", raw, pos)
        (count,) = struct.unpack_from("<Q", raw, pos + 4)
        pos += 12
        items: list[Any] = []
        for _ in range(count):
            item, pos = _read_value(raw, pos, elem)
            items.append(item)
        return (elem, items), pos
    raise ValueError(f"unsupported metadata type {vtype}")


@dataclass
class GGUFReadResult:
    version: int
    metadata: list[tuple[str, Any]]
    tensors: list[GGUFTensor]

    def metadata_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.metadata:
            out.setdefault(key, value)
        return out


def read_gguf(path: str | Path) -> GGUFReadResult:
    path = Path(path)
    raw = open(path, "rb").read()
    if raw[:4] != GGUF_MAGIC:
        raise ValueError(f"{path}: bad magic (not GGUF)")
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", raw, 4)
    if version != 3:
        raise ValueError(f"{path}: expected GGUF v3, got {version}")
    pos = 4 + 4 + 8 + 8  # 24-byte header

    metadata: list[tuple[str, Any]] = []
    for _ in range(n_kv):
        key, pos = _read_string(raw, pos)
        (vtype,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        value, pos = _read_value(raw, pos, vtype)
        metadata.append((key, value))

    infos: list[tuple[str, int, tuple[int, ...], int]] = []
    for _ in range(n_tensors):
        name, pos = _read_string(raw, pos)
        (ndims,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        dims_rev = tuple(
            struct.unpack_from("<Q", raw, pos + 8 * i)[0] for i in range(ndims)
        )
        pos += 8 * ndims
        dims = tuple(reversed(dims_rev))  # spec stores dims in reverse order
        (ggml_type,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        (offset,) = struct.unpack_from("<Q", raw, pos)
        pos += 8
        infos.append((name, ggml_type, dims, offset))

    tensors: list[GGUFTensor] = []
    data_start = (pos + 32 - 1) // 32 * 32
    for i, (name, ggml_type, dims, offset) in enumerate(infos):
        # offsets are relative to the aligned data-section start
        start = data_start + offset
        if i + 1 < len(infos):
            end = data_start + infos[i + 1][3]
        else:
            end = len(raw)
        blob = raw[start:end]
        tensors.append(GGUFTensor(name, ggml_type, dims, start, blob))

    return GGUFReadResult(version, metadata, tensors)


def inspect_gguf(path: str | Path) -> str:
    result = read_gguf(path)
    md = result.metadata_dict()
    names = ", ".join(t.name for t in result.tensors)
    return (
        f"GGUF ok: header offset 24, version {result.version}, "
        f"{len(result.tensors)} tensors, {len(result.metadata)} metadata keys\n"
        f"  metadata: {', '.join(md.keys())}\n"
        f"  tensors: {names}"
    )


def npz_round_trip(gguf: str | Path, npz: str | Path) -> bool:
    """Mirror the Kaggle-phase proof: gguf dequant == npz tensor."""
    result = read_gguf(gguf)
    data = np.load(npz, allow_pickle=False)
    if "logit_projection" not in data.files:
        return False
    expected = np.asarray(data["logit_projection"], dtype=np.float32)
    ours = result.tensors[0].dequant() if result.tensors else None
    if ours is None or ours.shape != expected.shape:
        return False
    return bool(np.allclose(ours, expected, atol=1e-3))
