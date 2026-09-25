"""Feather v1 — compare-benchmark tests.

These tests exercise the multi-file runner in kaggle/scripts/ without ever
importing torch, psutil, matplotlib or codecarbon (all optional in the
runner). They must stay fast: the whole batch stays well under 20s on CI so
the main/minimum_versions targets (numpy-only) never regress.
"""

from __future__ import annotations

from feather_v1 import FeatherV1Model
from kaggle.scripts import kaggle_benchmark as kb

MEDIUM_BINDINGS = (
    "avx512_wht",
    "avx2_wht",
    "avx_wht",
    "neon_wht",
    "scalar_wht",
)


def test_benchmark_config_contract_small():
    cfg = kb.benchmark_config(small=True)
    assert cfg.dim == 64
    assert cfg.seq_len == 128
    assert cfg.vocab_size == 4096
    assert 0 < cfg.ram_budget_gb <= 0.6


def test_benchmark_config_contract_medium():
    cfg = kb.benchmark_config(small=False)
    assert cfg.dim == 384
    assert cfg.seq_len == 512
    assert cfg.vocab_size == 4096
    assert cfg.dim * cfg.seq_len < 1 << 20


def test_measured_small_contract():
    cfg = kb.benchmark_config(small=True)
    model = FeatherV1Model(cfg)
    m = kb.benchmark_model(model, rows=8, reps=2)
    assert m["tok_per_sec"] > 0
    assert m["joules_per_1k"] >= 0
    assert m["ram_mb"] >= 0
    assert m["kernel_binding"] in MEDIUM_BINDINGS
    assert m["dim"] == 64
    assert m["config_seed"] == int(cfg.seed)


def test_measured_medium_contract():
    cfg = kb.benchmark_config(small=False)
    model = FeatherV1Model(cfg)
    m = kb.benchmark_model(model, rows=4, reps=2)
    assert m["tok_per_sec"] > 0
    assert m["dim"] == 384
    assert m["joules_per_1k"] >= 0
    assert m["kernel_hv"] >= 1024


def test_measured_row_contract():
    cfg = kb.benchmark_config(small=True)
    m = kb.benchmark_model(FeatherV1Model(cfg), rows=4, reps=1)
    kernel = {"binding": "avx2_wht", "expected_tok_per_sec": "45-60 tok/s"}
    row = kb._measured(m, kernel, "Feather v1 THIS PC (measured)")
    assert row["kind"] == "measured"
    assert "bulk" in row["speed"]
    assert row["mem_saving"] == "512x (2KB)"
    assert row["momr"] == "147x"
    assert row["cost"] == "$0 existing laptop"


def test_baselines_plus_measured_row_in_comparison():
    m = kb.benchmark_model(
        FeatherV1Model(kb.benchmark_config(small=True)), rows=4, reps=1
    )
    rows = kb.compare_rows(m)
    names = [r["name"] for r in rows]
    for expected in (
        "Transformer 7B GPU",
        "Transformer 7B CPU",
        "BitNet 100B CPU",
        "Phi-4 Mini 3.8B CPU",
        "Attention 512x384",
        "LSTM 384",
        "Feather v1 i7-12700 12C CPU",
        "Feather v1 THIS PC (measured)",
    ):
        assert expected in names, expected
    transformer = next(r for r in rows if r["name"] == "Transformer 7B GPU")
    assert transformer["energy_j_1k"] == 2.8
    assert transformer["cost"] == "$25k H100"


def test_energy_and_savings_estimates():
    assert kb.joules_per_1k_estimate({"binding": "avx512_wht"}) == 0.028
    assert kb.joules_per_1k_estimate({"binding": "avx2_wht"}) == 0.05
    assert kb.joules_per_1k_estimate({"binding": "avx_wht"}) == 0.08
    savings = kb.derived_report()
    assert savings["p_adic_vs_attention"]["mem_saving_x"] == 512.0
    assert savings["p_adic_vs_attention"]["ops_saving_x"] == round(
        kb.ATTN_SCORES / kb.PADIC_OPS, 1
    )
    assert savings["momr"]["feather_base_102m"] == 147


def test_chart_series_has_six_charts():
    series = kb.chart_data()
    assert list(series) == [
        "speed",
        "energy",
        "memory_saving",
        "ops_saving",
        "momr",
        "context",
    ]
    speed_labels = [e[0] for e in series["speed"]]
    assert "Feather v1 i7 CPU" in speed_labels
    assert "Feather i5-3337U" in speed_labels
    assert "Transformer 7B CPU" in speed_labels
    assert "iPhone" not in " ".join(speed_labels)
    assert any("3337" in label for label in speed_labels)
    momr_values = {e[0]: e[1] for e in series["momr"]}
    assert momr_values["Feather i7 CPU"] == 147.0
    assert momr_values["Feather Agent"] == 20.0
    context_values = [e[1] for e in series["context"]]
    assert 1000000 in context_values


def test_math_spot_checks_dry_from_utils():
    checks = kb.math_spot_checks()
    assert checks["lstm_cos_fails"] == -0.05
    assert checks["exp_decay_0_9_511"] < 1e-20
    assert checks["fractional_retention_w511"] > 0
    assert checks["fractional_vs_exp_x"] > 1
    assert checks["tt_rank4_g1_shape"] == [64, 4]
    assert abs(checks["fwht_norm_1024"] - 32.0) < 1.0
    assert checks["sinkhorn_rowstd"] < 1.0

    def _row(name):
        return name in checks

    assert _row("padic_d_0_64")
    assert _row("byte_tokens")
    assert _row("tropical_min_0_mults")
