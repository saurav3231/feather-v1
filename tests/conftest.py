"""Shared pytest fixtures for Feather v1 tests (DRY: no duplicated setup)."""

from __future__ import annotations

import numpy as np
import pytest

from feather_v1 import FeatherV1Config, FeatherV1Model


def make_config(**overrides) -> FeatherV1Config:
    defaults = dict(
        dim=64,
        seq_len=512,
        chunk_size=32,
        num_chunks=16,
        hypervector_dim=1024,
        alpha=0.7,
        k_frac=32,
        k_frac_long=128,
        tt_rank=4,
        tau=0.1,
        p_adic_p=2,
        seed=42,
        vocab_size=128,
    )
    defaults.update(overrides)
    return FeatherV1Config(**defaults)


def marker_sequence(seq_len=512, dim=64, magnitude=10.0):
    """Long-range recall task: distinctive marker at pos 0, noisy query at end."""
    rng = np.random.default_rng(42)
    marker = rng.standard_normal(dim)
    marker = marker / np.linalg.norm(marker) * magnitude
    seq = rng.standard_normal((seq_len, dim)) * 0.1
    seq[0] = marker
    query = marker + rng.standard_normal(dim) * 0.05
    query = query / np.linalg.norm(query) * magnitude
    return marker, seq, query


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def model():
    return FeatherV1Model(make_config())


@pytest.fixture
def marker():
    return marker_sequence()
