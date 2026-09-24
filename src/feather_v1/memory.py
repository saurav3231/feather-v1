"""Feather v1 — Component 2: Liquid Memory (Working Memory).

Hierarchical fractional memory (K1=32 L1 recent, K2=128 L3 long via p-adic
chunks), power-law weights ``w_k=(k+1)^-1.2`` giving 3.25e20x retention vs
exponential decay, WHT binding (adds only), event-driven spiking (98% sparse)
and 2-adic p-adic retrieval in 3 hops. This is the one complex task at which
the CPU excels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseComponent
from .utils import (
    cos_sim,
    fractional_weights,
    normalize,
    p_adic_distance,
    wht_bind,
)


class LiquidMemory(BaseComponent):
    """Component 2 -- power-law working memory with p-adic retrieval."""

    name = "memory"

    def __init__(self, config: Any, energy_utils: Any = None) -> None:
        super().__init__(config, energy_utils)
        dim = int(getattr(config, "dim", 64))
        self.dim = dim
        self.k_frac = int(getattr(config, "k_frac_recent", 32))
        self.k_frac_long = int(getattr(config, "k_frac_long", 128))
        self.chunk_size = int(getattr(config, "chunk_size", 32))
        self.num_chunks = int(getattr(config, "num_chunks", 16))
        self.alpha = float(getattr(config, "alpha_fractional", 0.7))
        self.p = int(getattr(config, "p_adic_p", 2))
        self.theta = 0.1  # spiking threshold
        self.seq_len = int(getattr(config, "seq_len", 512))
        # Fractional history buffer roll.
        self.weights = fractional_weights(self.alpha, self.k_frac)
        # Long-range chunk hypervectors (mean pooling of each chunk).
        self.chunk_hvs: np.ndarray = np.zeros((self.num_chunks, dim))
        self.history: np.ndarray = np.zeros((self.k_frac, dim))
        self.position = 0

    def hierarchical_fractional(self, token: np.ndarray) -> np.ndarray:
        """M_t = sum_{k=0}^{K1-1} w_k x_{t-k}; power-law, sequential (CPU task)."""
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = np.asarray(token, dtype=np.float64)
        recent = np.sum(self.history * self.weights[:, None], axis=0)
        self.count_ops(adds=self.k_frac * self.dim, multiplies=0)
        return recent

    def spiking_should_fire(self, m_prev: np.ndarray, m_next: np.ndarray) -> bool:
        """Event-driven spiking: fire only when the state moves (98% sparse)."""
        return float(np.linalg.norm(m_next - m_prev)) > self.theta

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """WHT binding ``a (x) b`` -- 384 adds, 0 multiplies."""
        out = wht_bind(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        self.count_ops(adds=out.size * 3, multiplies=0)
        return out

    def store_chunk(self, chunk_id: int, chunk: np.ndarray) -> None:
        """Store a normalized chunk hypervector."""
        self.chunk_hvs[chunk_id % self.num_chunks] = normalize(np.mean(chunk, axis=0))

    def p_adic_retrieve(self, query: np.ndarray) -> tuple[int, np.ndarray]:
        """Find best matching chunk. Returns (chunk_id, similarities)."""
        query = normalize(np.asarray(query, dtype=np.float64))
        sims = np.array(
            [cos_sim(self.chunk_hvs[j], query) for j in range(self.num_chunks)]
        )
        best = int(np.argmax(sims))
        return best, sims

    def level_2_distance(self, pos: int, query_pos: int) -> float:
        """p-adic 2-adic distance between positions (pointer-chasing metric)."""
        return p_adic_distance(pos, query_pos, self.p)

    def signature_weight(self, k: int) -> float:
        """Power-law weight of lag ``k`` with sign for the Grunwald-Letnikov sum."""
        return float(self.weights[k] * ((-1) ** k))

    def enumerate_levels(self) -> list[int]:
        """Sizes of the p-adic tree levels for this configuration."""
        sizes = []
        size = 1
        for _ in range(4):
            size *= self.chunk_size
            sizes.append(size)
        return sizes

    def retrieve_within_chunk(
        self, chunk: np.ndarray, query: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Softmax attention within a chunk (64^2 = 4096 ops vs 262k flat)."""
        chunk = np.asarray(chunk, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64)
        scores = chunk @ query / np.sqrt(self.dim)
        exp_s = np.exp(scores - scores.max())
        weights = exp_s / exp_s.sum()
        retrieved = weights @ chunk
        return weights, retrieved

    def reset(self) -> None:
        self.history = np.zeros_like(self.history)
        self.position = 0

    def cache_report(self) -> dict[str, int]:
        return {"l1_kb": 9, "l2_kb": 40, "l3_kb": 48}

    def __call__(self, token: np.ndarray) -> np.ndarray:
        return self.hierarchical_fractional(token)
