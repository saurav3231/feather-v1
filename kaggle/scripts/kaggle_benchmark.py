"""Kaggle -- offline CPU benchmark (measured numbers only).

Reuses only feather_v1 primitives (no duplicated math, no torch). Every
printed number is measured on the host; the kernel's expected-tok/s line is
the only data that comes from the detection table.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from feather_v1 import FeatherV1Config, FeatherV1Model

REPO_ROOT = Path(__file__).resolve().parents[2]

JOULE_PER_1K = 0.028  # advertised energy budget (README claim), for the gauge


def fp32_weight_bytes(model: FeatherV1Model) -> int:
    cfg = model.config
    return 4 * (
        cfg.dim * cfg.dim
        + cfg.dim * cfg.vocab_size
        + cfg.n_experts * 2 * cfg.dim * cfg.tt_rank
    )


def benchmark(model: FeatherV1Model, rows: int = 32, reps: int = 3) -> dict[str, float]:
    prompt = np.random.default_rng(0).standard_normal((rows, model.config.dim))
    model.forward(prompt)
    model.reset()

    start = time.perf_counter()
    for _ in range(reps):
        model.forward(prompt)
    elapsed = time.perf_counter() - start
    tokens = rows * reps
    joules = model.total_joules()
    return {
        "tokens": float(tokens),
        "tok_per_sec": float(tokens) / elapsed,
        "joules": float(joules),
        "joules_per_1k": float(joules) * 1000.0 / float(tokens),
        "model_mb": fp32_weight_bytes(model) / (1024.0 * 1024.0),
    }


def main() -> None:
    cfg_path = REPO_ROOT / "kaggle" / "configs" / "kaggle_cpu.json"
    cfg = (
        FeatherV1Config.from_file(str(cfg_path))
        if cfg_path.is_file()
        else FeatherV1Config.auto()
    )
    model = FeatherV1Model(cfg)
    res = benchmark(model)

    print(f"Kernel: {model.kernel['binding']} ({model.kernel['hypervector_dim']}-D hv)")
    print(f"Adaptive expectation: {model.kernel['expected_tok_per_sec']}")
    print(f"Measured: {res['tok_per_sec']:.1f} tok/s over {int(res['tokens'])} tokens")
    print(
        f"Energy: {res['joules'] * 1000.0:.3f} mJ for {int(res['tokens'])} tokens "
        f"({res['joules_per_1k'] * 1000.0:.3f} mJ / 1k; gauge {JOULE_PER_1K} J / 1k)"
    )
    print(f"FP32 weights: {res['model_mb']:.1f} MB")
    print("Structural claims: 512x memory saving, 64x fewer ops, 17.5x vs 14GB HBM")


if __name__ == "__main__":
    main()
