"""Feather v1 — Kaggle model/deployment tests (small + large scale).

Ensures the mechanisms the Kaggle notebooks rely on actually hold: the
final_output / cache_info / kernel keys, offline no-torch inference, the
byte-level tokenizer, energy accounting, K-FAC natural gradient, weight
save/load round-trip, and the 512x memory-saving shape guarantee.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import feather_v1
from feather_v1 import FeatherV1Config, FeatherV1Model
from feather_v1.utils import byte_decode, byte_tokenize, kfac_apply
from tests.conftest import make_config

REPO_ROOT = Path(feather_v1.__file__).resolve().parent.parent.parent
KAGGLE_CONFIGS = sorted((REPO_ROOT / "kaggle" / "configs").glob("*.json"))


@pytest.fixture
def model():
    return FeatherV1Model(make_config())


def test_forward_small_scale_returns_all_keys(model):
    rng = np.random.default_rng(1)
    out = model.forward(rng.standard_normal((8, 64)))
    for key in (
        "sensory",
        "memory_state",
        "knowledge",
        "reasoned",
        "protected",
        "drafts",
        "entropy_gate_exited",
        "final_output",
        "cache_info",
        "kernel",
        "token",
    ):
        assert key in out
    assert out["final_output"].shape == (1, 64)
    assert isinstance(out["token"], int)
    assert len(out["cache_info"]) == 6
    assert out["kernel"]["threads"] >= 1


def test_forward_large_scale_402_vocab384():
    cfg = make_config(
        dim=384,
        seq_len=512,
        chunk_size=32,
        num_chunks=16,
        vocab_size=4096,
    )
    m = FeatherV1Model(cfg)
    seq = np.random.default_rng(2).standard_normal((512, 384))
    out = m.forward(seq[: cfg.chunk_size])
    assert out["final_output"].shape == (1, 384)
    assert out["final_output"].shape[1] == cfg.dim
    assert np.all(np.isfinite(out["memory_state"]))
    assert out["kernel"]["hypervector_dim"] >= 512


def test_generate_offline_loop(model):
    prompt = np.zeros((4, model.config.dim))
    gen = model.generate(prompt, steps=6)
    assert gen.shape[0] <= model.config.seq_len
    assert gen.shape[1] == model.config.dim


def test_byte_tokenizer_roundtrip():
    text = "Feather v1 नेपाली + English morphology"
    ids = byte_tokenize(text)
    assert ids.min() >= 0 and ids.max() <= 255
    assert byte_decode(ids) == text
    assert byte_decode(byte_tokenize("")) == ""


def test_kfac_apply_natural_gradient_step():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 16))
    x = rng.standard_normal((32, 16))
    a_fac = (x.T @ x) / 32 + 1e-3 * np.eye(16)
    g = rng.standard_normal((16, 16)) * 0.1
    g_fac = (g.T @ g) / 32 + 1e-3 * np.eye(16)
    w2 = kfac_apply(w, a_fac, g_fac, lr=0.1)
    assert w.shape == w2.shape
    assert np.all(np.isfinite(w2))


def test_energy_accounted_and_tiny(model):
    model.forward(np.zeros((8, 64)))
    total = model.total_joules()
    assert total >= 0.0
    assert total < 1e-5
    report = model.energy_report()
    assert isinstance(report, dict)
    assert all(isinstance(v, float) for v in report.values())


def test_cache_reports_within_l1_l2_scope(model):
    for comp in (
        model.sensory,
        model.memory,
        model.knowledge,
        model.reasoning,
        model.governor,
        model.generation,
    ):
        rep = comp.cache_report()
        assert isinstance(rep, dict)
        assert rep.get("l1_kb", 0) <= 4096


def test_memory_saving_512x_and_weight_budget():
    cfg = make_config(seq_len=512, chunk_size=32, num_chunks=16)
    factor = cfg.seq_len**2 / cfg.chunk_size**2
    assert factor >= 200
    m = FeatherV1Model(cfg)
    assert m.config.ram_budget_gb <= 2.0
    weight_bytes = 4 * (
        cfg.dim * cfg.dim
        + cfg.dim * cfg.vocab_size
        + cfg.n_experts * 2 * cfg.dim * cfg.tt_rank
    )
    assert weight_bytes < 128 * 2**20


def test_save_load_weights_roundtrip(model, tmp_path):
    model.forward(np.zeros((4, 64)))
    path = str(tmp_path / "m.npz")
    model.save_weights(path)
    m2 = FeatherV1Model.from_weights(path)
    out1 = model.forward(np.zeros((4, 64)))
    out2 = m2.forward(np.zeros((4, 64)))
    assert np.allclose(out1["memory_state"], out2["memory_state"])
    assert np.allclose(out1["final_output"], out2["final_output"])
    assert np.allclose(m2._logit_projection, model._logit_projection)


def test_core_never_imports_torch():
    code = (
        "import sys; import numpy as np\n"
        "from feather_v1 import FeatherV1Config, FeatherV1Model\n"
        "from feather_v1.utils import byte_tokenize\n"
        "cfg = FeatherV1Config(dim=64, seq_len=64, chunk_size=8, num_chunks=8, vocab_size=128)\n"
        "m = FeatherV1Model(cfg)\n"
        "out = m.forward(np.zeros((2, 64)))\n"
        "assert out['final_output'].shape == (1, 64)\n"
        "assert byte_tokenize('x').tolist() == [120]\n"
        "assert 'torch' not in sys.modules\n"
    )
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, **env},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_expected_tok_per_sec_is_parseable(model):
    claim = model.kernel["expected_tok_per_sec"]
    assert "tok/s" in claim
    assert any(ch.isdigit() for ch in claim)


def test_kaggle_config_rebuilds_forward():
    assert len(list(KAGGLE_CONFIGS)) == 5
    first = json.loads(KAGGLE_CONFIGS[0].read_text(encoding="utf-8"))
    assert "feather_v1_config" in first
    cfg = FeatherV1Config.from_file(str(KAGGLE_CONFIGS[0]))
    assert cfg.dim >= 64
    m = FeatherV1Model(cfg)
    rng = np.random.default_rng(3)
    out = m.forward(rng.standard_normal((4, cfg.dim)))
    assert out["final_output"].shape == (1, cfg.dim)
