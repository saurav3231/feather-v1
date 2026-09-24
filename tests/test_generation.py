"""Tests for Component 6: Generative Evolution."""

import numpy as np

from feather_v1 import GenerativeEvolution
from feather_v1.utils import adaptive_draft_len
from tests.conftest import make_config


def _gen(config=None, **overrides):
    return GenerativeEvolution(config or make_config(vocab_size=128, **overrides))


def test_adaptive_draft_len_low_entropy(config):
    assert adaptive_draft_len(0.2) == 4  # easy -> 4 tokens
    assert adaptive_draft_len(0.9) == 8  # hard -> 8 tokens


def test_jacobi_converges_fast(config):
    gen = _gen(config)
    n = 4
    calls = []

    def logits_fn(k, candidates):
        calls.append(k)
        # trivial unigram model: always predict a fixed token
        out = np.zeros(128)
        out[7] = 10.0
        return out

    drafts = gen.speculative_generate(logits_fn, entropy=0.2, draft=np.zeros(n))
    assert drafts.shape == (n,)
    assert np.all(drafts == 7)
    assert gen.average_jacobi_iters <= 4


def test_sheaf_consistency(config):
    gen = _gen(config)
    ident = np.eye(4)
    # identical local sections restrict trivially (identity maps) -> consistent
    same = np.tile(np.arange(4.0), (3, 1))
    assert gen.sheaf_consistency(same, ident) is True
    # genuinely different drafts are NOT consistent under identity maps
    rn = np.random.default_rng(0)
    diff = rn.standard_normal((3, 4))
    assert gen.sheaf_consistency(diff, ident) is False
    # sections compatible on the overlap (shared first coordinate) are consistent
    proj = np.zeros((1, 4))
    proj[0, 0] = 1.0
    compatible = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 11.0, 12.0, 13.0]])
    assert gen.sheaf_consistency(compatible, proj) is True


def test_godel_edit_only_if_proven(config):
    gen = _gen(config)
    assert gen.godel_propose_edit(utility_old=1.0, utility_new=1.2) is True
    assert gen.godel_propose_edit(utility_old=1.2, utility_new=1.19) is False


def test_concept_probing_ranks(config):
    gen = _gen(config)
    concept = {"nepali": np.arange(8.0), "english": np.arange(8.0) + 1.0}
    ranked = gen.concept_probe(np.arange(8.0) + 0.5, concept)
    assert ranked[0] == "english"


def test_p_adic_tree_path(config):
    gen = _gen(config)
    path = gen.p_adic_tree_path(2, chunk_size=2)
    assert path == [0, 1]


def test_generation_matches_model_api(model):
    out = model.generate(
        np.random.default_rng(0).standard_normal((8, model.config.dim))
    )
    assert out.shape[1] == model.config.dim
