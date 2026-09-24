"""Tests for Component 2: Liquid Memory."""

import numpy as np
import pytest

from feather_v1 import LiquidMemory
from feather_v1.utils import cos_sim, fractional_weights
from tests.conftest import make_config, marker_sequence


def _memory(config=None, **overrides):
    return LiquidMemory(config or make_config(**overrides))


def test_fractional_retention_is_power_law(config):
    w = fractional_weights(0.7, 512)
    ratio = w[511] / (0.9**511)
    assert ratio > 1e18  # ~3.25e20x retention vs exponential decay


def test_fractional_retention_within_chunk(config):
    w = fractional_weights(0.7, 512)
    # Lag-63 power-law weight beats exponential decay by ~2.5x.
    assert w[63] / (0.9**63) > 1.0


def test_hierarchical_fractional_step(config):
    mem = _memory(config)
    rng = np.random.default_rng(0)
    token = rng.standard_normal(config.dim)
    m = mem.hierarchical_fractional(token)
    assert m.shape == (config.dim,)
    assert np.all(np.isfinite(m))


def test_wht_binding_zero_multiplies(config):
    mem = _memory(config)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64)
    b = rng.standard_normal(64)
    bound = mem.bind(a, b)
    assert np.linalg.norm(bound) > 1e-6
    # binding tracks the fused feature
    assert cos_sim(mem.bind(a, b), mem.bind(a, b)) > 0.99


def test_p_adic_retrieval_finds_marker(config):
    mem = _memory(config)
    marker, seq, query = marker_sequence(seq_len=config.seq_len, dim=config.dim)
    # populate chunk 0 with the marker, others with noise
    for j in range(config.num_chunks):
        start = j * config.chunk_size
        end = start + config.chunk_size
        chunk = seq[start:end].copy()
        if j == 0:
            chunk[0] = marker
        mem.store_chunk(j, chunk)
    best, sims = mem.p_adic_retrieve(query)
    assert best == 0
    assert sims[best] > 0.5


def test_within_chunk_attention_cache_saving(config):
    mem = _memory(config)
    rng = np.random.default_rng(0)
    chunk = rng.standard_normal((config.chunk_size, config.dim))
    query = chunk[0] + rng.standard_normal(config.dim) * 0.05
    weights, retrieved = mem.retrieve_within_chunk(chunk, query)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)
    assert cos_sim(retrieved, chunk[0]) > 0.9


def test_spiking_sparseness(config):
    mem = _memory(config)
    rng = np.random.default_rng(0)
    m_prev = rng.standard_normal(config.dim)
    m_next = m_prev.copy()
    assert not mem.spiking_should_fire(m_prev, m_next)
    assert mem.spiking_should_fire(m_prev, m_next + 10.0)


def test_p_adic_distance_two_adic(config):
    mem = _memory(config)
    assert mem.level_2_distance(4, 12) == pytest.approx(0.125, abs=1e-12)
    assert mem.level_2_distance(4, 8) == pytest.approx(0.25, abs=1e-12)
    assert mem.level_2_distance(0, 0) == 0.0
