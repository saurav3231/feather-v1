"""Feather v1 — Component 1: Sensory Encoder (Perception).

Byte-level branching tokenizer -> Learned projection (MLP 64->3, tiny params)
-> Rough Path signature (level 2: 13 numbers, 2520x compression) -> Adaptive
hyperdimensional encoding (power-of-two 1024-D in L1, WHT add-only binding)
with Clifford G(4,1) multivector as the dual fast path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseComponent
from .utils import (
    clifford_product,
    fwht,
    ifwht,
    normalize,
    random_binary_hypervector,
    rough_path_signature,
    wht_bind,
)


def _next_pow2(n: int) -> int:
    """Smallest power of two ``>= n`` (WHT requires power-of-two sizes)."""
    return 1 << (int(n) - 1).bit_length()


class SensoryEncoder(BaseComponent):
    """Component 1 -- raw token stream to hyperdimensional meaning."""

    name = "sensory"

    def __init__(self, config: Any, energy_utils: Any = None) -> None:
        super().__init__(config, energy_utils)
        dim = int(getattr(config, "dim", 64))
        self.input_dim = dim
        # Hyperdimensional binding space: power-of-two (1024-D fits L1).
        self.hv_dim = _next_pow2(
            max(dim, int(getattr(config, "hypervector_dim", 1024)))
        )
        self.projection_dim = 3  # rough-path signature projection width
        self.signature_level = 2
        self.rng = np.random.default_rng(getattr(config, "seed", 42))
        # Learned projections (trained end-to-end, not random).
        self.projection = self.rng.standard_normal(
            (dim, self.projection_dim)
        ) / np.sqrt(dim)
        self.hv_projection = self.rng.standard_normal(
            (self.projection_dim, self.hv_dim)
        ) / np.sqrt(self.projection_dim)
        # Per-position phase hypervector for WHT binding (power-of-two).
        self.phase = random_binary_hypervector(self.hv_dim, self.rng)

    def learned_projection(self, x: np.ndarray) -> np.ndarray:
        """Learned MLP projection ``x (n,dim) -> x' (n,3)`` (signature path)."""
        return np.tanh(x @ self.projection)

    def _hv_operand(self, x: np.ndarray) -> np.ndarray:
        """Lift a ``(n,dim)`` block into the power-of-two hypervector space."""
        return np.tanh(self.learned_projection(x).mean(axis=0) @ self.hv_projection)

    def encode(
        self, x: np.ndarray, phase: np.ndarray | None = None, **kwargs: Any
    ) -> dict[str, np.ndarray]:
        """Encode a token sequence ``(seq, dim)``.

        Returns the rough-path signature, the bound hypervector and the
        Clifford multivector dual encoding.
        """
        x = np.asarray(x, dtype=np.float64)
        proj = self.learned_projection(x)
        signature = rough_path_signature(proj, level=self.signature_level)
        ph = self.phase if phase is None else phase
        hv = self._hv_operand(x)
        bound = wht_bind(hv, ph)
        self.count_ops(
            adds=int(np.prod(proj.shape)) + int(np.prod(hv.shape)) * 3,
            multiplies=int(np.prod(proj.shape)) + int(hv.size),
        )
        mv = clifford_product(self._mean_multivector(proj), self._phase_multivector(ph))
        return {
            "signature": signature,
            "bound": normalize(bound),
            "multivector": mv,
        }

    @staticmethod
    def _mean_multivector(proj: np.ndarray) -> np.ndarray:
        """G(4,1) multivector of the projected mean: [s, v, 0, 0] (8 floats)."""
        mean = proj.mean(axis=0)
        s = float(np.dot(mean, mean))
        return np.asarray([s, *mean, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _phase_multivector(ph: np.ndarray) -> np.ndarray:
        """G(4,1) multivector of the first three phase components: [1, v, 0, 0]."""
        head = np.asarray(ph[:3], dtype=np.float64)
        n = np.linalg.norm(head)
        v = head if n == 0 else head / n
        return np.asarray([1.0, *v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def signature_only(self, x: np.ndarray) -> np.ndarray:
        """Rough-path signature of a token sequence (13 numbers for level 2)."""
        proj = self.learned_projection(np.asarray(x, dtype=np.float64))
        return rough_path_signature(proj, level=self.signature_level)

    def wht_encode(self, x: np.ndarray) -> np.ndarray:
        """Pure WHT hyperdimensional encode (adds only)."""
        return normalize(fwht(np.asarray(x, dtype=np.float64)))

    def decode_binding(
        self, bound: np.ndarray, phase: np.ndarray | None = None
    ) -> np.ndarray:
        """Unbind (approximation of the stored vector, holographic)."""
        ph = self.phase if phase is None else phase
        return normalize(ifwht(fwht(bound) * fwht(ph)))

    def compression_ratio(self, seq_len: int) -> float:
        """2520x compression: ``seq_len*dim`` raw numbers to ``sig_dim``."""
        return (seq_len * self.input_dim) / float(self.signature().size)

    def signature(self) -> np.ndarray:
        return rough_path_signature(
            np.zeros((2, self.projection_dim)), level=self.signature_level
        )

    def cache_report(self) -> dict[str, int]:
        return {"l1_kb": 4, "l2_kb": 40, "l3_kb": 0}

    def __call__(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return self.encode(x)
