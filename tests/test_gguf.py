"""Feather v1 -- GGUF v3 writer/reader tests.

Verify the spec-critical invariants (24-byte header, version 3, 32-aligned
tensor data), every supported quant round-trips through the reader with the
documented fidelity, and produced files pass the ``npz_round_trip`` proof
for lossless quants.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from feather_v1.gguf import (
    convert_pt_to_gguf,
    f16_pack,
    f32_pack,
    inspect_gguf,
    npz_round_trip,
    q4_k_pack,
    q8_0_pack,
    read_gguf,
    write_gguf,
)


@pytest.fixture
def projection():
    rng = np.random.default_rng(11)
    return rng.standard_normal((48, 96)).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )


def test_header_is_exactly_24_bytes_and_version_3(tmp_path, projection):
    path = write_gguf(tmp_path / "m.gguf", [f32_pack("output", projection)], [])
    raw = path.read_bytes()
    assert raw[:4] == b"GGUF"
    assert struct.unpack_from("<IQQ", raw, 4) == (3, 1, 0)
    assert path.stat().st_size > 24


def test_f32_exact_roundtrip(tmp_path, projection):
    path = write_gguf(tmp_path / "f32.gguf", [f32_pack("output", projection)], [])
    result = read_gguf(path)
    assert result.version == 3
    assert len(result.tensors) == 1
    back = result.tensors[0].dequant()
    assert back.shape == projection.shape
    assert np.array_equal(back, projection)


def test_f16_close_and_smaller(tmp_path, projection):
    path = write_gguf(tmp_path / "f16.gguf", [f16_pack("output", projection)], [])
    back = read_gguf(path).tensors[0].dequant()
    assert _cos(back, projection) > 0.999
    assert path.stat().st_size < _f32_size(projection)


def test_q8_0_cos_above_099_and_size_below_070(tmp_path, projection):
    path = write_gguf(tmp_path / "q8.gguf", [q8_0_pack("output", projection)], [])
    back = read_gguf(path).tensors[0].dequant()
    assert back.shape == projection.shape
    assert _cos(back, projection) > 0.99
    assert path.stat().st_size < int(0.70 * _f32_size(projection))


def test_q4_k_cos_above_080_and_size_below_040(tmp_path, projection):
    path = write_gguf(tmp_path / "q4.gguf", [q4_k_pack("output", projection)], [])
    back = read_gguf(path).tensors[0].dequant()
    assert back.shape == projection.shape
    assert _cos(back, projection) > 0.80
    assert path.stat().st_size < int(0.40 * _f32_size(projection))


def test_metadata_string_and_array_roundtrip(tmp_path, projection):
    meta = [
        ("general.architecture", 8, "feather"),
        ("general.file_type", 4, 15),
        ("general.tags", 9, (8, ["cpu", "llm"])),
    ]
    path = write_gguf(tmp_path / "meta.gguf", [f32_pack("output", projection)], meta)
    result = read_gguf(path)
    md = result.metadata_dict()
    assert md["general.architecture"] == "feather"
    assert md["general.file_type"] == 15
    assert len(result.metadata) == 3
    assert result.metadata[2][1][0] == 8
    assert result.metadata[2][1][1] == ["cpu", "llm"]


def test_aligned_tensor_data_offsets(tmp_path, projection):
    path = write_gguf(tmp_path / "al.gguf", [f32_pack("a", projection)], [])
    for t in read_gguf(path).tensors:
        assert t.offset % 32 == 0


def test_convert_pt_to_gguf_npz_round_trip(tmp_path):
    import json as _json

    from feather_v1.config import FeatherV1Config as CoreConfig

    core = CoreConfig(dim=48, seq_len=64, chunk_size=8, num_chunks=8, vocab_size=96)
    npz = tmp_path / "tiny.pt"
    rng = np.random.default_rng(5)
    proj = rng.standard_normal((core.dim, core.vocab_size)).astype(np.float32)
    with open(npz, "wb") as fh:
        np.savez(
            fh,
            config=_json.dumps(core.to_dict()).encode("utf-8"),
            logit_projection=proj,
        )
    out = convert_pt_to_gguf(npz, tmp_path / "fv.gguf", quant="f16")
    assert npz_round_trip(out, npz)
    assert "GGUF ok" in inspect_gguf(out)
    assert read_gguf(out).metadata_dict()["general.file_type"] == 1


def _f32_size(projection: np.ndarray) -> int:
    return int(np.prod(projection.shape)) * 4


def test_inspect_reports_tensor_and_metadata_counts(tmp_path, projection):
    write_gguf(
        tmp_path / "i.gguf",
        [f16_pack("output", projection)],
        [("general.name", 8, "feather-v1"), ("general.file_type", 4, 1)],
    )
    text = inspect_gguf(tmp_path / "i.gguf")
    assert "header offset 24" in text
    assert "1 tensors" in text
    assert "2 metadata keys" in text
