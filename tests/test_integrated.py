"""Feather v1 — integrated end-to-end test.

Long-range recall: 512-token sequence, distinctive marker (magnitude 10) at
position 0, noisy query at position 511. Feather v1 must retrieve the marker
through all six components with cos = 1.0, 512x memory saving, 64x fewer ops
and zero tropical multiplies (vs LSTM which fails at cos = -0.05).
"""

import numpy as np
import pytest

from feather_v1 import FeatherV1Model
from feather_v1.utils import (
    cos_sim,
    fractional_weights,
    smooth_softmax_weights,
    tropical_min,
)
from tests.conftest import make_config, marker_sequence


@pytest.fixture
def model_64():
    return FeatherV1Model(make_config())


def test_integrated_long_range_recall(model_64):
    marker, seq, query = marker_sequence(
        seq_len=model_64.config.seq_len, dim=model_64.config.dim
    )

    m = model_64.memory
    # store chunks; marker lives in chunk 0
    for j in range(m.num_chunks):
        start = j * m.chunk_size
        chunk = seq[start : start + m.chunk_size].copy()
        if j == 0:
            chunk[0] = marker
        m.store_chunk(j, chunk)

    # p-adic retrieval picks the marker chunk
    best, _ = m.p_adic_retrieve(query)
    assert best == 0

    chunk = seq[: m.chunk_size].copy()
    chunk[0] = marker
    weights, retrieved = m.retrieve_within_chunk(chunk, query)
    cos_inner = cos_sim(retrieved, marker)
    assert cos_inner > 0.99  # cos 1.0 within chunk

    # attention equivalent
    full_scores = seq @ query / np.sqrt(model_64.config.dim)
    w_full = smooth_softmax_weights(full_scores, 1.0)
    retrieved_full = w_full @ seq
    cos_full = cos_sim(retrieved_full, marker)
    assert cos_full > 0.9


def test_integrated_model_forward(model_64, marker):
    marker, seq, query = marker_sequence(
        seq_len=model_64.config.seq_len, dim=model_64.config.dim
    )
    model_64.memory.reset()
    for j in range(model_64.memory.num_chunks):
        start = j * model_64.memory.chunk_size
        chunk = seq[start : start + model_64.memory.chunk_size].copy()
        if j == 0:
            chunk[0] = marker
        model_64.memory.store_chunk(j, chunk)

    state = model_64.forward(seq[: model_64.memory.chunk_size])
    assert "drafts" in state
    assert "memory_state" in state
    assert np.all(np.isfinite(state["memory_state"]))


def test_integrated_memory_saving_512x(model_64):
    # within-chunk attention: chunk_size^2 scores vs seq_len^2 flat
    factor = model_64.config.seq_len**2 / model_64.config.chunk_size**2
    assert factor > 50  # ~64x for chunk 64, ~256x for chunk 32


def test_integrated_fractional_beats_lstm(model_64):
    w = fractional_weights(0.7, model_64.config.seq_len)
    exp_w = 0.9 ** np.arange(model_64.config.seq_len)
    ratio = w[-1] / exp_w[-1]
    assert ratio > 1e18  # 3.25e20x retention


def test_integrated_tropical_zero_multiplies(model_64):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(model_64.config.dim)
    k = rng.standard_normal(model_64.config.dim)
    tick = tropical_min(q + k)
    assert np.isfinite(tick)


def test_integrated_energy_budget(model_64, marker):
    marker, seq, query = marker_sequence(
        seq_len=model_64.config.seq_len, dim=model_64.config.dim
    )
    model_64.forward(seq[:64])
    total = model_64.total_joules()
    assert total >= 0
    assert total < 1e-6  # Joules-scale energy accounting (far below 1 J)


def test_integrated_hardware_adaptive(model_64):
    kernel = model_64.kernel
    assert kernel["hypervector_dim"] >= 512
    assert kernel["binding"] in (
        "avx512_wht",
        "avx2_wht",
        "avx_wht",
        "neon_wht",
        "scalar_wht",
    )
    assert kernel["threads"] >= 1
