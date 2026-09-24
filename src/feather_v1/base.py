"""Feather v1 — base component.

Every one of the six components inherits :class:`BaseComponent` for common
logic: configuration, the adaptive hardware kernel, energy accounting and
cache sizing. No component re-implements this plumbing.
"""

from __future__ import annotations

from typing import Any

from .hardware import get_best_kernel
from .utils import cos_sim


class BaseComponent:
    """Common behaviour shared by all Feather v1 components."""

    name = "base"

    def __init__(
        self,
        config: Any,
        energy_utils: Any | None = None,
    ) -> None:
        self.config = config
        self.kernel: dict[str, Any] = get_best_kernel()
        # Optional codecarbon tracker for per-component Joules logging.
        self.energy_tracker = energy_utils
        self._joules: float = 0.0
        self.total_ops: int = 0
        self.total_multiplies: int = 0

    # -- core API (overridden by subclasses) --------------------------------
    def encode(self, x: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # -- energy -------------------------------------------------------------
    def record_energy(self, joules: float) -> None:
        self._joules += float(joules)
        if self.energy_tracker is not None:
            try:
                self.energy_tracker.record(self.name, joules)
            except Exception:
                pass

    @property
    def joules(self) -> float:
        return self._joules

    def reset_energy(self) -> None:
        self._joules = 0.0

    def count_ops(self, adds: int = 0, multiplies: int = 0) -> None:
        self.total_ops += int(adds) + int(multiplies)
        self.total_multiplies += int(multiplies)

    @property
    def multiplies(self) -> int:
        return self.total_multiplies

    # -- cache/layout help --------------------------------------------------
    def cache_report(self) -> dict[str, int]:
        """Rough kB footprint of this component's working set."""
        return {"l1_kb": 0, "l2_kb": 0, "l3_kb": 0}

    # -- utility ------------------------------------------------------------
    @staticmethod
    def similarity(a: Any, b: Any) -> float:
        return cos_sim(a, b)
