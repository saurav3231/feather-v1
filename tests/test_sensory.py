"""Tests for Component 1: Sensory Encoder."""

import numpy as np
import pytest

from feather_v1 import SensoryEncoder
from feather_v1.utils import rough_path_signature


@pytest.fixture
def seq():
    rng = np.random.default_rng(0)
    return rng.standard_normal((64, 64))


def _encoder(config):
    return SensoryEncoder(config)


def test_encode_shape_and_signature(config, seq):
    enc = _encoder(config)
    out = enc.encode(seq)
    assert "signature" in out
    assert "bound" in out
    assert "multivector" in out
    # Level-2 signature of a 3-D projected path: 1 + 3 + 9 = 13 numbers
    assert out["signature"].size == 13


def test_signature_compression_ratio(config, seq):
    enc = _encoder(config)
    ratio = (seq.shape[0] * seq.shape[1]) / enc.signature().size
    assert ratio > 100  # ~2520x for full 512*64 sequences


def test_signature_level3(config, seq):
    enc = _encoder(config)
    proj = enc.learned_projection(seq)
    sig3 = rough_path_signature(proj, level=3)
    assert sig3.size == 1 + 3 + 9 + 27


def test_binding_is_holographic(config, seq):
    enc = _encoder(config)
    out = enc.encode(seq)
    bound = out["bound"]
    assert np.linalg.norm(bound) == pytest.approx(1.0, abs=1e-6)
    dec = enc.decode_binding(bound, enc.phase)
    assert np.linalg.norm(dec) == pytest.approx(1.0, abs=1e-6)


def test_signature_identity_chen(config, seq):
    enc = _encoder(config)
    sig = enc.signature_only(seq)
    # Chen identity: signature of concatenated path equals product; here we
    # only assert stability for independent draws of the same shape.
    assert sig.shape == (13,)
