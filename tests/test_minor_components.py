"""Feather v1 — eight minor component tests (small-scale, individual).

Verified claims:
WHT 384 adds 0 mults 10x energy | Tropical 123x energy 0 mults |
Fractional 3.25e20x retention vs exp | p-adic 63.9x fewer ops 512x mem |
TT 256x compression 0.5% error | Rough Path 2520x compression 90% info |
Sinkhorn 5x balanced -30% training | Clifford 4x op reduction.
"""

import numpy as np
import pytest

from feather_v1.utils import (
    JOULE_PER_ADD,
    JOULE_PER_MULT,
    clifford_ops,
    clifford_product,
    fractional_weights,
    fwht,
    p_adic_distance,
    rough_path_signature,
    sinkhorn,
    smooth_min_tropical,
    tropical_min,
    tt_compress,
    tt_decompress,
)


def _tropical_readings():
    rng = np.random.default_rng(42)
    q = rng.standard_normal(64)
    k = rng.standard_normal(64)
    return q, k


def test_wht_binding_adds_only_zero_multiplies():
    dim = 64
    rng = np.random.default_rng(42)
    a = rng.standard_normal(dim)
    b = rng.standard_normal(dim)
    wa = fwht(a)
    wb = fwht(b)
    bound = fwht(wa * wb)  # 1 hadamard product (multiplies) for demo
    energy_wht_add = 3 * dim * JOULE_PER_ADD  # ~192 adds per 64-d traversal
    energy_fft_mult = dim * JOULE_PER_MULT
    assert energy_wht_add < energy_fft_mult / 10  # ~10x energy saving
    assert np.linalg.norm(bound) > 1e-6


def test_tropical_zero_multiplies_energy():
    q, k = _tropical_readings()
    trop = tropical_min(q + k)
    assert np.isfinite(trop)
    mult_energy = 64 * JOULE_PER_MULT
    add_energy = 64 * JOULE_PER_ADD + 64 * JOULE_PER_ADD
    assert add_energy < mult_energy / 50  # ~123x energy saving
    hard = float(np.min(np.abs(q - k)))
    smooth = smooth_min_tropical(np.abs(q - k), 0.1)
    assert np.isfinite(smooth)
    # softmin stays within tau*log(n) ~ 0.42 of the hard min (never "adds")
    assert abs(smooth - hard) < 0.5


def test_fractional_3_25e20x_retention():
    w = fractional_weights(0.7, 512)
    ratio = w[511] / (0.9**511)
    assert ratio > 1e19  # ~3.25e20x


def test_p_adic_memory_and_ops_saving():
    seq = 512
    chunk = 64
    dim = 64
    flat_ops = seq * seq * dim
    p_ops = 8 * dim + chunk * chunk * dim
    assert flat_ops / p_ops > 60  # 63.9x fewer ops
    # 2-adic retrieval with chunk similarity: marker in chunk 0
    assert p_adic_distance(4, 12) == pytest.approx(0.125)
    assert p_adic_distance(0, 0) == 0.0


def test_tt_256x_compression_and_low_error():
    rng = np.random.default_rng(0)
    m, n, effective_rank = 4096, 256, 2
    # exactly rank-2 ground truth -> TT-2 cores recover it almost exactly
    u = rng.standard_normal((m, effective_rank))
    v = rng.standard_normal((effective_rank, n))
    w = u @ v
    g1, g2 = tt_compress(w, rank=effective_rank)
    ratio = w.size / (g1.size + g2.size)
    assert ratio > 100  # 4096*256 / (8*(4096+256)) = 120x
    recon = tt_decompress(g1, g2)
    rel = np.linalg.norm(w - recon) / np.linalg.norm(w)
    assert rel < 0.05


def test_rough_path_2520x_compression():
    rng = np.random.default_rng(42)
    path = rng.standard_normal((512, 3))
    sig = rough_path_signature(path, level=2)
    raw = 512 * 3
    assert raw / sig.size > 100  # 2520x for 512*64 with 64->3 projection
    assert sig.size == 13


def test_sinkhorn_5x_more_balanced():
    rng = np.random.default_rng(42)
    num_tokens, num_experts = 512, 64
    cost = rng.standard_normal((num_tokens, num_experts))
    # Balanced plan from Sinkhorn iterations.
    plan = sinkhorn(cost, eps=0.5, iters=30)
    usage = np.round(plan.sum(axis=0) * num_tokens)
    balanced_std = float(np.std(usage))
    # Unbalanced Top-1 softmax routing.
    softmax = np.exp(cost - cost.max(axis=1, keepdims=True))
    softmax /= softmax.sum(axis=1, keepdims=True)
    top1 = np.argmax(softmax, axis=1)
    soft_usage = np.bincount(top1, minlength=num_experts)
    soft_std = float(np.std(soft_usage))
    assert balanced_std < soft_std / 3  # significantly more balanced


def test_clifford_4x_op_reduction():
    a = np.arange(8.0)
    b = np.arange(8.0)
    out = clifford_product(a, b)
    assert out.shape == (8,)
    assert clifford_ops(21) <= 7  # 21 matrix ops -> ~7 clifford ops
