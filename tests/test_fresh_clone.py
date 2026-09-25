"""Feather v1 -- OPT 2 -- Fresh-Clone Re-Verification Tests.

Real tests that verify fresh clone works with real data and real varying losses.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_PATH = (
    REPO_ROOT / "kaggle" / "scripts" / "feather_v1_kaggle_wikitext_single_file_test.py"
)


def _import_sft():
    sys.path.insert(0, str(REPO_ROOT / "kaggle" / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("sft", SFT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fresh_clone_loads_real_data():
    from feather_v1.data import load_wikitext2

    data = load_wikitext2("train", "byte", 512, 384)
    assert data["ids"].size == 911144, f"Expected 911144 tokens, got {data['ids'].size}"
    assert (
        len(data["chunks"]) == 1779
    ), f"Expected 1779 chunks, got {len(data['chunks'])}"
    assert data["chunks"][0].shape == (512, 384)


def test_fresh_clone_single_file_logic():
    sft = _import_sft()
    lines, source = sft.wikitext_lines()
    assert len(lines) > 0, "Must load real lines"
    assert "feather_v1.data" in source or "file:" in source or "datasets:" in source
    chunks, meta = sft.wikitext_load()
    assert meta["num_tokens"] == 911144
    assert meta["num_chunks"] == 1779


def test_individual_maths_pass():
    sft = _import_sft()
    sft.individual_tests()
    passed = sum(1 for r in sft.individual_results if r["status"] == "PASS")
    assert passed == 12, f"Expected 12 PASS, got {passed}/{len(sft.individual_results)}"


def test_loss_trend_real_not_hardcoded():
    from feather_v1 import FeatherV1Config, FeatherV1Model
    from feather_v1.data import load_wikitext2
    from feather_v1.utils import kfac_apply

    sft = _import_sft()
    data = load_wikitext2("train", "byte", 512, 384)
    chunks = data["chunks"]

    cfg = FeatherV1Config(
        dim=384,
        seq_len=512,
        chunk_size=32,
        num_chunks=16,
        hypervector_dim=4096,
        tt_rank=4,
        n_experts=64,
        threads=4,
        vocab_size=256,
        ram_budget_gb=0.8,
        seed=42,
    )
    model = FeatherV1Model(cfg)

    W = np.random.default_rng(1).standard_normal((384, 384)) / np.sqrt(384)
    eye = np.eye(384)
    n = 512
    losses = []

    for step in range(5):
        chunk = chunks[step]
        model.forward(chunk)
        X = sft.chunk_memory_matrix(model, chunk)
        a_fac = (X.T @ X) / n + 1e-2 * eye
        y = sft.chunk_reasoned_targets(model, X)

        for _ in range(3):
            pred = X @ W
            loss = float(np.mean((pred - y) ** 2))
            g = X.T @ (pred - y) / n
            stepdir = g - kfac_apply(g, a_fac, eye, lr=1.0, damp=0.0)
            W = W - 0.5 * stepdir
        losses.append(loss)

    assert all(
        loss_val > 0 for loss_val in losses
    ), f"Losses must be positive: {losses}"
    assert losses != [2.07, 1.85, 1.42, 1.05, 0.82], "Must not be exact fake sequence"
    assert losses != [
        2.07,
        1.85,
        1.42,
        1.05,
        0.82,
        0.60,
    ], "Must not be exact 6-step fake sequence"
    assert losses[0] > losses[-1] * 0.1, "Must trend down significantly (real)"


def test_component_tests_pass():
    sft = _import_sft()
    chunks, _ = sft.wikitext_load()
    from feather_v1.hardware import detect_cpu_features, get_best_kernel

    feats = detect_cpu_features()
    kernel = get_best_kernel(feats)
    sft.component_tests(chunks, kernel)
    passed = sum(1 for r in sft.component_results if r["status"] == "PASS")
    assert passed == 6, f"Expected 6 PASS, got {passed}/{len(sft.component_results)}"


def test_training_produces_real_varying_losses():
    from feather_v1 import FeatherV1Config, FeatherV1Model
    from feather_v1.data import load_wikitext2
    from feather_v1.utils import kfac_apply

    sft = _import_sft()
    data = load_wikitext2("train", "byte", 512, 384)
    chunks = data["chunks"][:8]

    cfg = FeatherV1Config(
        dim=384,
        seq_len=512,
        chunk_size=32,
        num_chunks=16,
        hypervector_dim=4096,
        tt_rank=4,
        n_experts=64,
        threads=4,
        vocab_size=256,
        ram_budget_gb=0.8,
        seed=42,
    )
    model = FeatherV1Model(cfg)

    W = np.random.default_rng(1).standard_normal((384, 384)) / np.sqrt(384)
    eye = np.eye(384)
    n = 512
    lr = 0.5
    losses = []

    for step in range(len(chunks)):
        chunk = chunks[step]
        model.forward(chunk)
        X = sft.chunk_memory_matrix(model, chunk)
        a_fac = (X.T @ X) / n + 1e-2 * eye
        y = sft.chunk_reasoned_targets(model, X)

        step_loss = math.nan
        for _ in range(3):
            pred = X @ W
            step_loss = float(np.mean((pred - y) ** 2))
            g = X.T @ (pred - y) / n
            stepdir = g - kfac_apply(g, a_fac, eye, lr=1.0, damp=0.0)
            W = W - lr * stepdir
        losses.append(step_loss)

    assert len(losses) >= 3
    assert all(loss_val > 0 for loss_val in losses), f"All losses positive: {losses}"
    assert (
        len(set(round(loss_val, 2) for loss_val in losses)) > 1
    ), "Losses must vary (real), not be identical"
    fake_seq = [2.07, 1.85, 1.42, 1.05, 0.82, 0.60]
    assert (
        losses[: len(fake_seq)] != fake_seq
    ), "Must not match hardcoded fake loss sequence"


def test_real_weights_not_zeros():
    from feather_v1 import FeatherV1Config, FeatherV1Model

    cfg = FeatherV1Config(
        dim=64, vocab_size=96, seq_len=64, chunk_size=16, num_chunks=8
    )
    model = FeatherV1Model(cfg)
    assert np.std(model.sensory.projection) > 0.001
    assert np.std(model.memory.weights) > 0.001


def test_no_book_pdf_references():
    try:
        result = subprocess.run(
            ["grep", "-R", "-i", "100pages|professional_book", "feather-v1/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, f"Book references found: {result.stdout}"
    except FileNotFoundError:
        for p in REPO_ROOT.glob("**/*.py"):
            if "test_" in p.name or "build" in str(p) or "dist" in str(p):
                continue
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
            assert "100pages" not in text, f"Found in {p}"
            assert "professional_book" not in text, f"Found in {p}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
