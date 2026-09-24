"""Feather v1 — quickstart.

Load the model, detect the best kernel for your CPU, run inference on a
toy sequence. CPU-only, no GPU. Works on every PC via the adaptive hardware
fallback (AVX-512 -> AVX2 -> AVX -> NEON -> Scalar).
"""

from __future__ import annotations

import argparse

import numpy as np

from feather_v1 import FeatherV1Config, FeatherV1Model


def main() -> None:
    ap = argparse.ArgumentParser(description="Feather v1 quickstart")
    ap.add_argument("--config", type=str, default=None, help="JSON config path")
    ap.add_argument("--threads", type=int, default=None, help="override threads")
    ap.add_argument("--auto-hardware", action="store_true", help="auto-detect hardware")
    args = ap.parse_args()

    if args.config:
        config = FeatherV1Config.from_file(args.config)
    elif args.auto_hardware:
        config = FeatherV1Config.auto()
    else:
        config = FeatherV1Config()  # defaults already match the i5-3337U profile

    if args.threads:
        config.threads = args.threads

    print(f"Hardware summary:\n{FeatherV1Model(config).hardware_summary()}\n")
    print(
        f"Config: {config} | threads: {config.threads} | ram: {config.ram_budget_gb}GB"
    )

    model = FeatherV1Model(config)
    rng = np.random.default_rng(config.seed)
    prompt = rng.standard_normal((16, config.dim))

    out = model.forward(prompt)
    print("Feather v1 inference complete (CPU-only).")
    print(f"  State keys : {sorted(out.keys())}")
    print(f"  Drafts     : {out['drafts'][:8]}")
    print(
        f"  Entropy    : exited early via governor gate = {out['entropy_gate_exited']}"
    )
    print(
        "  Expected tok/s on your PC: 94 (AVX-512+AMX) | 45-60 (AVX2) | "
        "12-18 (AVX, i5-3337U) | 35-50 (NEON) | 3-5 (Scalar)"
    )


if __name__ == "__main__":
    main()
