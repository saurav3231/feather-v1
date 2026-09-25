"""Feather v1 -- OPT 3 -- REAL weights tests.

The OPT 3 deliverable replaces the synthetic weight bundle with weights that
were actually trained on the REAL WikiText-2 corpus (911144 tokens, shipped
in-repo).  These tests use the real trained artifacts:

* ``checkpoints/feather-v1-{5M,20M,100M}/feather-v1-real.pt``
* ``dist/feather-v1-{size}.{q4_k_m,f16}.gguf``

Full-scale tests are skipped when the artifacts are absent (CI builds wheels
without running Kaggle training); the fast tests always run and prove the
corpus + loader + real-GGUF metadata contract.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

import feather_v1
from feather_v1.data import load_wikitext2, wikitext_path
from feather_v1.gguf import npz_round_trip, read_gguf
from feather_v1.gguf.real import REAL_METADATA_KEYS, build_real_gguf

REPO_ROOT = Path(feather_v1.__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
DIST_DIR = REPO_ROOT / "dist"

SIZES = ["5M", "20M", "100M"]


def _has_artifacts() -> bool:
    return all(
        (CHECKPOINT_DIR / f"feather-v1-{s}" / "feather-v1-real.pt").is_file()
        for s in SIZES
    )


def _has_ggufs() -> bool:
    return all((DIST_DIR / f"feather-v1-{s}.q4_k_m.gguf").is_file() for s in SIZES)


# ---------------------------------------------------------------------------
# Always-on fast tests: real corpus shipping + loader guarantees
# ---------------------------------------------------------------------------
def test_real_corpus_ships_in_repo_not_fetched():
    path = wikitext_path("train")
    assert path.is_file()
    text = path.read_bytes()
    assert b"Valkyria Chronicles III" in text[:80]
    assert len(text) == 915142  # first 2000 nonempty lines, exact size
    assert b"wiki" not in text[:20]


def test_real_corpus_byte_tokens_exactly_911144():
    data = load_wikitext2("train", "byte", 512, 384)
    assert data["meta"]["num_tokens"] == 911144
    assert data["meta"]["vocab_size"] == 256
    assert data["meta"]["source"].endswith("wikitext-train-raw-v1.txt")
    assert len(data["chunks"]) > 1700
    assert data["chunks"][0].shape == (512, 384)


def test_real_corpus_char_tokens_exactly_96_vocab():
    data = load_wikitext2("train", "char", 512, 384)
    assert data["meta"]["vocab_size"] == 96
    assert data["meta"]["num_tokens"] > 900_000


# ---------------------------------------------------------------------------
# GGUF metadata contract (runs with or without real GGUF files)
# ---------------------------------------------------------------------------
def _real_scale_projection(seed, rows=64, cols=96):
    """Like a trained logit_projection: dense, nonzero, small-magnitude (std ~0.05)."""
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((rows, cols)).astype(np.float32) / np.sqrt(rows)
    return proj


def test_real_gguf_metadata_is_exactly_16_keys(tmp_path):
    proj = _real_scale_projection(7)
    cfg = json.dumps(
        {
            "dim": 64,
            "vocab_size": 96,
            "seq_len": 64,
            "num_chunks": 16,
            "seed": 42,
        }
    ).encode("utf-8")
    npz = tmp_path / "tiny-real.pt"
    with open(npz, "wb") as fh:
        np.savez(fh, config=cfg, logit_projection=proj)
    result = build_real_gguf(npz, tmp_path / "fv-q4.gguf", quant="q4_k_m", size="5M")
    assert result["n_metadata"] == REAL_METADATA_KEYS == 16
    assert result["n_tensors"] == 1
    gguf = read_gguf(tmp_path / "fv-q4.gguf")
    raw = (tmp_path / "fv-q4.gguf").read_bytes()
    assert raw[:4] == b"GGUF"
    assert struct.unpack_from("<IQQ", raw, 4) == (3, 1, 16)  # v3, 1 tensor, 16 metadata
    assert gguf.metadata_dict()["general.file_type"] == 15
    assert gguf.metadata_dict()["general.architecture"] == "feather"
    assert gguf.metadata_dict()["feather.vocab_size"] == 96
    # q4_k_m is lossy: strict round-trip is not required, only f16/f32 are.
    back = gguf.tensors[0].dequant()
    cos = float(
        np.dot(proj.ravel(), back.ravel())
        / (np.linalg.norm(proj) * np.linalg.norm(back) + 1e-12)
    )
    assert cos > 0.80


def test_real_gguf_rejects_zero_weights(tmp_path):
    npz = tmp_path / "zero.pt"
    with open(npz, "wb") as fh:
        np.savez(
            fh,
            config=json.dumps({"dim": 64, "vocab_size": 96}).encode("utf-8"),
            logit_projection=np.zeros((64, 96), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="all zeros"):
        build_real_gguf(npz, tmp_path / "zero.gguf", quant="f16", size="5M")


def test_real_gguf_f16_roundtrip_lossless(tmp_path):
    proj = _real_scale_projection(8)
    with open(tmp_path / "w.pt", "wb") as fh:
        np.savez(
            fh,
            config=json.dumps(
                {
                    "dim": 64,
                    "vocab_size": 96,
                    "seq_len": 64,
                    "num_chunks": 16,
                    "seed": 42,
                }
            ).encode("utf-8"),
            logit_projection=proj,
        )
    build_real_gguf(tmp_path / "w.pt", tmp_path / "fv-f16.gguf", quant="f16", size="5M")
    assert npz_round_trip(str(tmp_path / "fv-f16.gguf"), str(tmp_path / "w.pt"))


# ---------------------------------------------------------------------------
# Full-scale gates: require the real trained artifacts
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _has_ggufs(), reason="real GGUF artifacts not built (OPT 3 training)"
)
def test_real_ggufs_all_sizes_exist_and_have_right_shape():
    for size in SIZES:
        for quant in ("q4_k_m", "f16"):
            gguf = DIST_DIR / f"feather-v1-{size}.{quant}.gguf"
            assert gguf.is_file(), f"{gguf} missing"
            hdr = read_gguf(gguf)
            assert len(hdr.tensors) == 1
            assert len(hdr.metadata) == 16
            assert hdr.tensors[0].dequant().shape[1] >= 64


@pytest.mark.skipif(
    not _has_ggufs(), reason="real GGUF artifacts not built (OPT 3 training)"
)
def test_real_ggufs_round_trip_against_real_checkpoints():
    for size in SIZES:
        for quant in ("q4_k_m", "f16"):
            gguf = DIST_DIR / f"feather-v1-{size}.{quant}.gguf"
            pt = CHECKPOINT_DIR / f"feather-v1-{size}" / "feather-v1-real.pt"
            assert pt.is_file()
            if quant == "f16":
                assert npz_round_trip(str(gguf), str(pt))
            else:
                src = np.load(pt, allow_pickle=False)["logit_projection"]
                got = read_gguf(gguf).tensors[0].dequant().astype(np.float64)
                cos = float(
                    np.dot(src.ravel(), got.ravel())
                    / (
                        np.linalg.norm(src.ravel()) * np.linalg.norm(got.ravel())
                        + 1e-12
                    )
                )
                assert cos > 0.80, f"{size} {quant} cos={cos:.3f} < 0.80 (lossy quant)"
                assert not np.allclose(got, 0.0, atol=1e-8)  # real weights, not zeros


@pytest.mark.skipif(not _has_artifacts(), reason="real checkpoints not trained (OPT 3)")
def test_real_checkpoints_are_real_not_synthetic():
    for size in SIZES:
        pt = CHECKPOINT_DIR / f"feather-v1-{size}" / "feather-v1-real.pt"
        data = np.load(pt, allow_pickle=False)
        proj = data["logit_projection"]
        assert proj.ndim == 2
        assert float(np.std(proj)) > 0.0  # not zeros
        cfg = json.loads(bytes(data["config"]).decode("utf-8"))
        assert cfg["dim"] == proj.shape[0]
        assert cfg["vocab_size"] == proj.shape[1]


@pytest.mark.skipif(not _has_artifacts(), reason="real checkpoints not trained (OPT 3)")
def test_real_checkpoints_match_size_legends():
    legend = {"5M": 64, "20M": 384, "100M": 512}
    for size, dim in legend.items():
        pt = CHECKPOINT_DIR / f"feather-v1-{size}" / "feather-v1-real.pt"
        data = np.load(pt, allow_pickle=False)
        assert data["logit_projection"].shape[0] == dim
