"""Tests for Component 5: Homeostasis Governor."""

import numpy as np
import pytest

from feather_v1 import HomeostasisGovernor
from feather_v1.utils import krum_select, trimmed_mean
from tests.conftest import make_config


def _gov(config=None, **overrides):
    return HomeostasisGovernor(config or make_config(**overrides))


def test_entropy_gate_early_exit(config):
    gov = _gov(config)
    p_low = np.array([0.95, 0.05])
    p_high = np.array([0.25, 0.25, 0.25, 0.25])
    assert gov.entropy_gate(p_low) is True
    assert gov.entropy_gate(p_high) is False


def test_energy_budget_penalty(config):
    gov = _gov(config)
    assert gov.energy_budget_penalty(1.0) == pytest.approx(0.01)


def test_landauer_penalty(config):
    gov = _gov(config)
    assert gov.landauer_penalty(1) > 0


def test_byzantine_robust_aggregation(config):
    gov = _gov(config)
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((20, 8))
    poisoned = clean.copy()
    poisoned[0] = clean[0] + 1000.0
    result = gov.byzantine_robust_aggregate(poisoned, trim=0.25)
    clean_mean = clean.mean(axis=0)
    # naive mean is dragged toward the +1000 outlier in every column
    naive = poisoned.mean(axis=0)
    assert np.linalg.norm(naive - clean_mean) > 50.0
    # robust trimmed mean stays within sampling noise of the clean aggregate
    assert np.linalg.norm(result["trimmed_mean"] - clean_mean) < 1.0
    assert np.all(np.isfinite(result["krum"]))


def test_krum_select_ignores_outlier(config):
    rng = np.random.default_rng(0)
    grads = rng.standard_normal((8, 4))
    outlier = grads[0].copy() + 100.0
    grads[0] = outlier
    chosen = krum_select(grads, num_byzantine=1)
    # krum should NOT pick the outlier
    assert not np.allclose(chosen, outlier)


def test_trimmed_mean_removes_outliers(config):
    rng = np.random.default_rng(0)
    grads = rng.standard_normal((10, 4))
    grads[0] += 500.0
    tm = trimmed_mean(grads, trim=0.25)
    assert np.all(np.abs(tm) < 100.0)


def test_dp_laplace_privacy(config):
    gov = _gov(config)
    values = np.zeros((4, 4))
    noisy = gov.dp_noise(values)
    assert noisy.shape == (4, 4)
    assert np.abs(noisy).max() > 0  # noise added
    assert gov.dp_epsilon == 1.0


def test_free_energy(config):
    gov = _gov(config)
    f = gov.free_energy(energy=1.0, temperature=0.1, entropy=0.5, complexity=0.1)
    assert np.isfinite(f)


def test_record_ops_energy(config):
    gov = _gov(config)
    j = gov.record_ops_energy(adds=10, multiplies=2)
    assert j > 0
    assert gov.joules == j
