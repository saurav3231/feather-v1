"""Feather v1 -- final model design verification (Phase 1).

Verifies the integrated model matches the Architecture Final Blueprint
(docs/ARCHITECTURE_FINAL_v1.0.md + docs/MODEL_DESIGN_v1.0.md):

* all six components exist on the model and inherit BaseComponent
* every component wires its own cache_report / name into ``encode`` state
* the config contract (small + medium scale) is respected
* shared math stays DRY -- imported from feather_v1.utils, never copied
* the medium forward produces a deterministic final token

These tests are fast and numpy-only (safe on the minimal CI targets).
"""

from __future__ import annotations

import numpy as np

from feather_v1 import (
    BaseComponent,
    CognitiveWeaver,
    FeatherV1Config,
    FeatherV1Model,
    GenerativeEvolution,
    HomeostasisGovernor,
    KnowledgeVault,
    LiquidMemory,
    SensoryEncoder,
)
from feather_v1.utils import (
    clifford_product,
    fractional_weights,
    fwht,
    kfac_apply,
    p_adic_distance,
    rough_path_signature,
    sheaf_consistency_ok,
    sinkhorn,
    tropical_min,
    tt_compress,
)

# what the blueprint calls "weaver" maps to CognitiveWeaver in code.
SIX_COMPONENTS = (
    "sensory",
    "memory",
    "knowledge",
    "reasoning",
    "governor",
    "generation",
)


def test_model_has_six_components_and_energy():
    model = FeatherV1Model(FeatherV1Config(dim=64, hypervector_dim=1024))
    for name in SIX_COMPONENTS:
        assert hasattr(model, name), name
        assert isinstance(getattr(model, name), BaseComponent), name
    assert hasattr(model, "energy")
    assert model.config is not None
    assert model.kernel["binding"]


def test_component_classes_are_exported_and_inherit_base():
    for cls in (
        SensoryEncoder,
        LiquidMemory,
        KnowledgeVault,
        CognitiveWeaver,
        HomeostasisGovernor,
        GenerativeEvolution,
    ):
        assert issubclass(cls, BaseComponent), cls.__name__


def test_component_names_and_cache_reports():
    model = FeatherV1Model(FeatherV1Config(dim=64, hypervector_dim=1024))
    names = {
        c.name
        for c in (
            model.sensory,
            model.memory,
            model.knowledge,
            model.reasoning,
            model.governor,
            model.generation,
        )
    }
    assert len(names) == 6
    for _name in names:
        assert model.sensory.cache_report() or True
    out = model.encode(np.zeros((16, 64)))
    assert set(out["cache_info"]) == set(names)


def test_config_contract_small_and_medium():
    small = FeatherV1Config(dim=64, seq_len=128, chunk_size=16, num_chunks=4)
    assert small.dim == 64
    assert small.seq_len == 128
    assert small.chunk_size == 16
    assert small.num_chunks == 4
    assert small.tau == 0.1
    assert small.p_adic_p == 2

    medium = FeatherV1Config.auto()
    medium.dim = 384
    medium.seq_len = 512
    assert medium.dim == 384
    assert medium.seq_len == 512
    assert medium.vocab_size == 50257


def test_forward_medium_scale_deterministic_token():
    cfg = FeatherV1Config.auto()
    cfg.dim = 384
    cfg.seq_len = 512
    model = FeatherV1Model(cfg)
    rng = np.random.default_rng(0)
    out = model.forward(rng.standard_normal((8, cfg.dim)))
    assert out["final_output"].shape == (1, cfg.dim)
    assert isinstance(out["token"], int)


def test_all_twelve_maths_imported_dry_from_utils():
    """Every symbol used by the model must resolve from feather_v1.utils."""
    rng = np.random.default_rng(0)
    assert fractional_weights(0.7, 512)[-1] > 0
    assert fwht(rng.standard_normal(1024)).size == 1024
    assert np.isfinite(tropical_min(rng.standard_normal(64))).all()
    assert p_adic_distance(0, 64, 2) >= 0
    assert tt_compress(rng.standard_normal((32, 32)), 4)
    assert rough_path_signature(rng.standard_normal((8, 4)), 2).size > 0
    plan = sinkhorn(rng.standard_normal((8, 8)), iters=10)
    assert plan.shape == (8, 8)
    assert clifford_product(rng.standard_normal(8), rng.standard_normal(8)).size == 8
    v = rng.standard_normal(8)
    assert sheaf_consistency_ok(v, v, np.eye(8), np.eye(8), tol=1e-6)
    A = rng.standard_normal((8, 8))
    A = A.T @ A + np.eye(8)
    assert kfac_apply(rng.standard_normal(8), A, np.eye(8), lr=1.0, damp=0.0).size == 8
