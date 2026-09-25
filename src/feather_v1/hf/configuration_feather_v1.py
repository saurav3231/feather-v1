"""Feather v1 -- HuggingFace ``PretrainedConfig`` (py38 compatible).

Mirrors the numpy :class:`feather_v1.config.FeatherV1Config` ``to_dict()``
camelCase keys so ``config.json`` written here round-trips 1:1 with the core
config, and builds a core config via ``to_core()``.  ``transformers`` is an
optional import; the class definition only needs it when instantiated.
"""

from __future__ import annotations

from typing import Any

try:
    from transformers import PretrainedConfig  # type: ignore
except Exception:  # pragma: no cover - transformers optional
    PretrainedConfig = object  # type: ignore[assignment,misc]

from ..config import FeatherV1Config as CoreFeatherV1Config

# Keys shared by the core config.to_dict() and the HF config.json.
_FIELDS = (
    "architecture",
    "dim",
    "seq_len",
    "chunk_size",
    "num_chunks",
    "hypervector_dim",
    "alpha_fractional",
    "K_frac_recent",
    "K_frac_long",
    "TT_rank",
    "tau_tropical",
    "p_adic_p",
    "threads",
    "precision",
    "binding",
    "moe",
    "n_experts",
    "moe_top_k",
    "sinkhorn_eps",
    "sinkhorn_iters",
    "batch_size",
    "ram_budget_gb",
    "seed",
    "vocab_size",
)


class FeatherV1Config(PretrainedConfig):  # type: ignore[misc, valid-type]
    model_type = "feather"

    def __init__(self, **kwargs: Any) -> None:
        """Accept every core config key; unknown kwargs stay in config_dict."""
        overrides: dict[str, Any] = {}
        for key in list(kwargs):
            if key in _FIELDS:
                overrides[key] = kwargs.pop(key)
        self.architecture = (
            overrides.pop("architecture", "FeatherV1")
            if "architecture" in overrides
            else "FeatherV1"
        )
        self.dim = int(overrides.pop("dim", 384))
        self.seq_len = int(overrides.pop("seq_len", 512))
        self.chunk_size = int(overrides.pop("chunk_size", 32))
        self.num_chunks = int(overrides.pop("num_chunks", 16))
        self.hypervector_dim = int(overrides.pop("hypervector_dim", 1024))
        self.alpha_fractional = float(overrides.pop("alpha_fractional", 0.7))
        self.K_frac_recent = int(overrides.pop("K_frac_recent", 32))
        self.K_frac_long = int(overrides.pop("K_frac_long", 128))
        self.TT_rank = int(overrides.pop("TT_rank", 4))
        self.tau_tropical = float(overrides.pop("tau_tropical", 0.1))
        self.p_adic_p = int(overrides.pop("p_adic_p", 2))
        self.threads = int(overrides.pop("threads", 2))
        self.precision = str(overrides.pop("precision", "int8"))
        self.binding = str(overrides.pop("binding", "avx_wht"))
        self.moe = str(overrides.pop("moe", "avx_tropical_tt"))
        self.n_experts = int(overrides.pop("n_experts", 64))
        self.moe_top_k = int(overrides.pop("moe_top_k", 1))
        self.sinkhorn_eps = float(overrides.pop("sinkhorn_eps", 0.1))
        self.sinkhorn_iters = int(overrides.pop("sinkhorn_iters", 3))
        self.batch_size = int(overrides.pop("batch_size", 1))
        self.ram_budget_gb = float(overrides.pop("ram_budget_gb", 0.6))
        self.seed = int(overrides.pop("seed", 42))
        self.vocab_size = int(overrides.pop("vocab_size", 50257))
        super().__init__(**kwargs)

    def to_core(self) -> CoreFeatherV1Config:
        """Build the numpy runtime config this HF config describes."""
        return CoreFeatherV1Config(
            dim=self.dim,
            seq_len=self.seq_len,
            chunk_size=self.chunk_size,
            num_chunks=self.num_chunks,
            hypervector_dim=self.hypervector_dim,
            alpha=float(self.alpha_fractional),
            k_frac=int(self.K_frac_recent),
            k_frac_long=int(self.K_frac_long),
            tt_rank=int(self.TT_rank),
            tau=float(self.tau_tropical),
            p_adic_p=int(self.p_adic_p),
            threads=int(self.threads),
            precision=self.precision,
            binding=self.binding,
            moe=self.moe,
            n_experts=int(self.n_experts),
            moe_top_k=int(self.moe_top_k),
            sinkhorn_eps=float(self.sinkhorn_eps),
            sinkhorn_iters=int(self.sinkhorn_iters),
            batch_size=int(self.batch_size),
            ram_budget_gb=float(self.ram_budget_gb),
            seed=int(self.seed),
            vocab_size=int(self.vocab_size),
        )

    @classmethod
    def from_core(cls, core: CoreFeatherV1Config, **kwargs: Any) -> FeatherV1Config:
        """Wrap a numpy config; extra ``kwargs`` (e.g. pad_token_id) ride along."""
        data = core.to_dict()
        data.update(kwargs)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["model_type"] = self.model_type
        return data
