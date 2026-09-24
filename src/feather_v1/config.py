"""Feather v1 — configuration (py38 compatible)."""

from __future__ import annotations

from typing import Any

from .hardware import load_config


class FeatherV1Config:
    """Configuration for Feather v1 (mirrors ``configs/*.json``).

    Defaults follow the i5-3337U-optimized profile:
    dim 384, hypervector 1024-D (4KB L1), chunk 32 (8KB L1), K_frac 32 L1
    + 128 L3, TT rank 4 (8x), threads 2 physical, int8, avx_wht, 0.6GB RAM.
    """

    def __init__(
        self,
        dim: int = 384,
        seq_len: int = 512,
        chunk_size: int = 32,
        num_chunks: int = 16,
        hypervector_dim: int = 1024,
        alpha: float = 0.7,
        k_frac: int = 32,
        k_frac_long: int = 128,
        tt_rank: int = 4,
        tau: float = 0.1,
        p_adic_p: int = 2,
        threads: int = 2,
        precision: str = "int8",
        binding: str = "avx_wht",
        moe: str = "avx_tropical_tt",
        n_experts: int = 64,
        moe_top_k: int = 1,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 3,
        batch_size: int = 1,
        ram_budget_gb: float = 0.6,
        seed: int = 42,
        vocab_size: int = 50257,
    ) -> None:
        self.dim = dim
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.hypervector_dim = hypervector_dim
        self.alpha = alpha
        self.k_frac = k_frac
        self.k_frac_long = k_frac_long
        self.tt_rank = tt_rank
        self.tau = tau
        self.p_adic_p = p_adic_p
        self.threads = threads
        self.precision = precision
        self.binding = binding
        self.moe = moe
        self.n_experts = n_experts
        self.moe_top_k = moe_top_k
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.batch_size = batch_size
        self.ram_budget_gb = ram_budget_gb
        self.seed = seed
        self.vocab_size = vocab_size

    @property
    def alpha_fractional(self) -> float:
        return self.alpha

    @property
    def tau_tropical(self) -> float:
        return self.tau

    @property
    def p_adic_chunk(self) -> int:
        return self.chunk_size

    @property
    def k_frac_recent(self) -> int:
        return self.k_frac

    # -- io -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "seq_len": self.seq_len,
            "chunk_size": self.chunk_size,
            "num_chunks": self.num_chunks,
            "hypervector_dim": self.hypervector_dim,
            "alpha_fractional": self.alpha,
            "K_frac_recent": self.k_frac,
            "K_frac_long": self.k_frac_long,
            "TT_rank": self.tt_rank,
            "tau_tropical": self.tau,
            "p_adic_p": self.p_adic_p,
            "threads": self.threads,
            "precision": self.precision,
            "binding": self.binding,
            "moe": self.moe,
            "n_experts": self.n_experts,
            "moe_top_k": self.moe_top_k,
            "sinkhorn_eps": self.sinkhorn_eps,
            "sinkhorn_iters": self.sinkhorn_iters,
            "batch_size": self.batch_size,
            "ram_budget_gb": self.ram_budget_gb,
            "seed": self.seed,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatherV1Config:
        kernel = None

        def pick(key: str, fallback: Any) -> Any:
            v = data.get(key, fallback)
            if isinstance(v, str) and v.strip().lower() == "auto":
                nonlocal kernel
                if kernel is None:
                    from .hardware import get_best_kernel

                    kernel = get_best_kernel()
                hv = int(kernel["hypervector_dim"])
                if key == "hypervector_dim":
                    return hv
                if key == "chunk_size":
                    return 64 if hv >= 4096 else 32
                if key == "TT_rank":
                    return 16 if hv >= 10000 else 8 if hv >= 4096 else 4
                if key == "threads":
                    return int(kernel["threads"])
                if key == "binding":
                    return str(kernel["binding"])
                if key == "moe":
                    return str(kernel["moe"])
            return v

        return cls(
            dim=int(pick("dim", 384)),
            seq_len=int(pick("seq_len", 512)),
            chunk_size=int(pick("chunk_size", 32)),
            num_chunks=int(pick("num_chunks", 16)),
            hypervector_dim=int(pick("hypervector_dim", 1024)),
            alpha=float(pick("alpha_fractional", 0.7)),
            k_frac=int(pick("K_frac_recent", 32)),
            k_frac_long=int(pick("K_frac_long", 128)),
            tt_rank=int(pick("TT_rank", 4)),
            tau=float(pick("tau_tropical", 0.1)),
            p_adic_p=int(pick("p_adic_p", 2)),
            threads=int(pick("threads", 2)),
            precision=str(pick("precision", "int8")),
            binding=str(pick("binding", "avx_wht")),
            moe=str(pick("moe", "avx_tropical_tt")),
            n_experts=int(pick("n_experts", 64)),
            moe_top_k=int(pick("moe_top_k", 1)),
            sinkhorn_eps=float(pick("sinkhorn_eps", 0.1)),
            sinkhorn_iters=int(pick("sinkhorn_iters", 3)),
            batch_size=int(pick("batch_size", 1)),
            ram_budget_gb=float(pick("ram_budget_gb", 0.6)),
            seed=int(pick("seed", 42)),
            vocab_size=int(pick("vocab_size", 50257)),
        )

    @classmethod
    def from_file(cls, path: str) -> FeatherV1Config:
        """Load from a ``configs/*.json`` file (with hardware auto-merging)."""
        cfg_dict = load_config(path)
        return cls.from_dict(cfg_dict.get("feather_v1_config", {}))

    @classmethod
    def auto(cls, **overrides: Any) -> FeatherV1Config:
        """Build a hardware-adaptive config for the current machine.

        Mirrors ``configs/all_pcs.json``: hypervector dimension, chunk size,
        TT rank and threads adapt to the detected kernel.
        """
        from .hardware import get_best_kernel

        kernel = get_best_kernel()
        hv = int(kernel["hypervector_dim"])
        return cls(
            hypervector_dim=hv,
            chunk_size=64 if hv >= 4096 else 32,
            tt_rank=16 if hv >= 10000 else 8 if hv >= 4096 else 4,
            threads=int(kernel["threads"]),
            binding=str(kernel["binding"]),
            moe=str(kernel["moe"]),
            precision=str(kernel["precision"]),
            **overrides,
        )

    def __repr__(self) -> str:
        return (
            f"FeatherV1Config(dim={self.dim}, hv={self.hypervector_dim}, "
            f"chunk={self.chunk_size})"
        )
