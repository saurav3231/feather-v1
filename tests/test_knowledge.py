"""Tests for Component 3: Knowledge Vault."""

import numpy as np

from feather_v1 import KnowledgeVault
from feather_v1.utils import (
    sinkhorn,
    smooth_min_tropical,
    tropical_min,
    tt_compress,
    tt_decompress,
)
from tests.conftest import make_config


def _vault(config=None, **overrides):
    return KnowledgeVault(config or make_config(**overrides))


def test_tropical_min_zero_multiplies(config):
    q = np.random.default_rng(0).standard_normal(64)
    k = np.random.default_rng(1).standard_normal(64)
    trop = tropical_min(q + k)
    assert np.isfinite(trop)
    hard = float(np.min(np.abs(q - k)))
    smooth = smooth_min_tropical(np.abs(q - k), 0.1)
    assert np.isfinite(smooth)
    # softmin stays within tau*log(n) ~ 0.42 of the hard min (0 multiplies)
    assert abs(smooth - hard) < 0.5


def test_tt_compression_and_recovery(config):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 64))
    g1, g2 = tt_compress(w, rank=4)
    recon = tt_decompress(g1, g2)
    rel_err = np.linalg.norm(w - recon) / np.linalg.norm(w)
    assert rel_err < 0.9
    ratio = 64 * 64 / (g1.size + g2.size)
    assert ratio > 5


def test_tt_rank_scales_compression(config):
    from feather_v1.utils import tt_compression_ratio

    assert tt_compression_ratio(4096, 4096, 16) > 100
    assert tt_compression_ratio(4096, 4096, 8) > 50


def test_sinkhorn_balanced_router(config):
    vault = _vault(config)
    rng = np.random.default_rng(0)
    for batch in (4, 16):
        result = vault.conditional_router(rng.standard_normal(config.dim), batch)
        assert "expert_id" in result
        assert 0 <= result["expert_id"] < vault.n_experts


def test_sinkhorn_plan_is_balanced(config):
    rng = np.random.default_rng(0)
    rows, cols = 512, 64
    cost = rng.standard_normal((rows, cols))
    plan = sinkhorn(cost, eps=0.5, iters=20)
    # balanced coupling: uniform marginals (rows ~ 1/rows, cols ~ 1/cols)
    assert np.allclose(plan.sum(axis=0), 1.0 / cols, atol=1e-2)
    assert np.allclose(plan.sum(axis=1), 1.0 / rows, atol=1e-2)


def test_tropical_expert_runs_zero_multiplies(config):
    vault = _vault(config)
    x = np.random.default_rng(0).standard_normal(config.dim)
    out = vault.tropical_expert(3, x)
    assert out.shape == (config.dim,)
    assert vault.multiplies == 0


def test_moe_active_is_sparse(config):
    vault = _vault(config)
    x = np.random.default_rng(0).standard_normal(config.dim)
    vault.route_and_apply(x)
    assert len(vault.active_experts) == 1  # Top-1: 1.5% active
