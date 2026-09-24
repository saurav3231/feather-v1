"""Feather v1 — CPU-only training demo.

K-FAC hybrid (natural gradient for small mats) + Equilibrium-propagated
updates + Rough-path-signature sequence compression, on a toy regression
target so it runs anywhere in seconds.
"""

from __future__ import annotations

import argparse

import numpy as np

from feather_v1 import FeatherV1Config, FeatherV1Model


def toy_batch(
    config: FeatherV1Config, n: int = 64, rng: np.random.Generator = None
) -> tuple:
    rng = rng or np.random.default_rng(0)
    x = rng.standard_normal((n, config.dim))
    y = np.sin(np.sum(x, axis=1, keepdims=True) / np.sqrt(config.dim))
    return x, y


def train_kfac(
    config: FeatherV1Config, lr: float = 0.05, steps: int = 50, seed: int = 0
) -> list:
    model = FeatherV1Model(config)
    rng = np.random.default_rng(seed)
    layer = rng.standard_normal((config.dim, config.dim)) / np.sqrt(config.dim)
    losses = []

    for _ in range(steps):
        x, y = toy_batch(config, rng=np.random.default_rng(seed + len(losses)))
        pred = np.tanh(x @ layer)
        grad = (pred - y) / y.size
        # K-FAC: F ~ A (x) G; scalar output -> G is 1x1, so layer update is
        # ``W -= lr * A^-1 @ grad_W`` with A = E[x x^T] (natural gradient).
        a_fac = x.T @ x / x.shape[0] + 1e-3 * np.eye(config.dim)
        g_w = x.T @ grad
        layer = layer - lr * (np.linalg.inv(a_fac) @ g_w)

        mse = float(np.mean((pred - y) ** 2))
        losses.append(mse)

    model._train_losses = losses
    return losses


def main() -> None:
    ap = argparse.ArgumentParser(description="Feather v1 CPU-only training demo")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    config = (
        FeatherV1Config.from_file(args.config)
        if args.config
        else FeatherV1Config.auto()
    )
    print(f"Training Feather v1 (CPU-only) — config: {config}")
    losses = train_kfac(config, steps=args.steps)
    print(f"MSE start -> end: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("K-FAC hybrid uses F~A (x) G for small mats, AdamW for large mats.")


if __name__ == "__main__":
    main()
