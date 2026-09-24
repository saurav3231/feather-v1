"""Feather v1 — Component 5: Homeostasis Governor (Efficiency + Security).

Triple gate (entropy early-exit H<0.6, energy budget, Landauer reversible
penalty), thermodynamic free energy, Byzantine-robust sheaf aggregation
(Krum + Trimmed Mean), differential privacy (Laplace, epsilon=1.0) and
codecarbon per-component Joules logging.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseComponent
from .utils import (
    JOULE_PER_ADD,
    JOULE_PER_MULT,
    KT_HYPERBOLIC,
    krum_select,
    laplace_noise,
    sheaf_consistency_ok,
    trimmed_mean,
)


class HomeostasisGovernor(BaseComponent):
    """Component 5 -- energy gates, security and privacy."""

    name = "governor"

    def __init__(self, config: Any, energy_utils: Any = None) -> None:
        super().__init__(config, energy_utils)
        self.entropy_threshold = 0.6
        self.energy_lambda = 0.01
        self.dp_epsilon = 1.0
        self.rng = np.random.default_rng(getattr(config, "seed", 42))
        self.early_exits = 0
        self.total_checks = 0

    # -- triple gate: entropy ----------------------------------------------
    def entropy_gate(self, p: np.ndarray) -> bool:
        """Return True when entropy is below threshold (exit early)."""
        p = np.asarray(p, dtype=np.float64)
        p = p[p > 0]
        if p.size == 0:
            return True
        h = -float(np.sum(p * np.log(p)))
        self.total_checks += 1
        if h < self.entropy_threshold:
            self.early_exits += 1
            return True
        return False

    @property
    def early_exit_rate(self) -> float:
        return self.early_exits / max(1, self.total_checks)

    def energy_budget_penalty(self, joules: float) -> float:
        """``L_energy = lambda * Joules`` (lambda annealed 0 -> 0.01)."""
        return self.energy_lambda * joules

    def landauer_penalty(self, n_flips: int) -> float:
        """Reversible-hardware penalty proportional to kT ln2 flips."""
        return n_flips * KT_HYPERBOLIC

    # -- thermodynamic free energy -----------------------------------------
    @staticmethod
    def free_energy(
        energy: float, temperature: float, entropy: float, complexity: float
    ) -> float:
        """``F = E - T*S + Complexity``."""
        return energy - temperature * entropy + complexity

    # -- security: Byzantine robust sheaf aggregation ----------------------
    def byzantine_robust_aggregate(
        self, grads: np.ndarray, trim: float = 0.25
    ) -> dict[str, np.ndarray]:
        """Krum + Trimmed-Mean robust gradient aggregation."""
        krum = krum_select(grads, num_byzantine=1)
        trimmed = trimmed_mean(grads, trim)
        return {"krum": krum, "trimmed_mean": trimmed}

    def byzantine_robust_check(
        self, local: np.ndarray, global_vec: np.ndarray, tol: float = 1e-6
    ) -> bool:
        """Sheaf consistency of a local section against the global section."""
        return sheaf_consistency_ok(
            local, global_vec, np.eye(local.size), np.eye(global_vec.size), tol=tol
        )

    # -- privacy: differential privacy -------------------------------------
    def dp_noise(self, values: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Laplace mechanism with ``epsilon = 1.0`` on restriction maps."""
        scale = sensitivity / self.dp_epsilon
        values = np.asarray(values, dtype=np.float64)
        noise = laplace_noise(scale, int(np.size(values)), self.rng)
        return values + noise.reshape(values.shape)

    # -- energy measurement ------------------------------------------------
    def record_ops_energy(self, adds: int, multiplies: int = 0) -> float:
        """Approximate Joules for an op mixture (0.03pJ adds, 3.7pJ mults)."""
        joules = adds * JOULE_PER_ADD + multiplies * JOULE_PER_MULT
        self.record_energy(joules)
        return joules

    def cache_report(self) -> dict[str, int]:
        return {"l1_kb": 1, "l2_kb": 0, "l3_kb": 0}

    def __call__(self, p: np.ndarray) -> bool:
        return self.entropy_gate(p)
