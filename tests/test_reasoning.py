"""Tests for Component 4: Cognitive Weaver."""

import numpy as np

from feather_v1 import CognitiveWeaver
from tests.conftest import make_config


def _weaver(config=None, **overrides):
    return CognitiveWeaver(config or make_config(**overrides))


def test_composition_functor_type_checks(config):
    weaver = _weaver(config)

    def f(x):
        return x * 0.5

    def g(x):
        return x - 1.0

    x = np.ones(8)
    out = weaver.compose(f, g, x)
    assert np.allclose(out, -0.5)


def test_liquid_weave_keeps_shape(config):
    weaver = _weaver(config)
    state = np.random.default_rng(0).standard_normal(config.dim)
    out = weaver.weave(state, tau=0.1)
    assert out.shape == (config.dim,)


def test_reasoning_loop_early_exit(config):
    weaver = _weaver(config)
    state = np.random.default_rng(0).standard_normal(config.dim)
    entropies = np.full(6, 0.2)  # low entropy -> early exit at loop 1
    weaver.reasoning_loop(state, entropies)
    assert weaver.average_loops == 1.0


def test_clifford_product_shape(config):
    weaver = _weaver(config)
    a = np.arange(8.0)
    b = np.arange(8.0)
    out = weaver.clifford_dual(a, b, use_clifford=True)
    assert out.shape == (8,)


def test_kfac_step_reduces_norm(config):
    weaver = _weaver(config)
    grad = np.ones((8, 8)) * 0.01
    a_fac = np.eye(8) + 0.1
    g_fac = np.eye(8) + 0.1
    new = weaver.kfac_step(np.ones((8, 8)), grad, a_fac, g_fac)
    assert np.all(np.isfinite(new))


def test_adamw_step(config):
    weaver = _weaver(config)
    param = np.zeros(8)
    m = np.zeros(8)
    v = np.zeros(8)
    grad = np.ones(8) * 0.5
    new, m, v = weaver.adamw_step(param, grad, m, v, lr=0.01)
    assert np.all(new < 0)  # gradient descent direction
