"""Kaggle -- host and kernel detection (bare CPU, no torch).

Runs anywhere: on a Kaggle runtime it prints the Kaggle facts (runtime type,
RAM, CPU flags, CUDA, internet) in addition to the adaptive CPU kernel. This
is the first cell of every Feather v1 notebook.
"""

from __future__ import annotations

from pathlib import Path

from feather_v1 import FeatherV1Config, FeatherV1Model
from feather_v1.hardware import kaggle_env, summary

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    print(summary())

    env = kaggle_env()
    if env["cuda_devices"]:
        for dev in env["cuda_devices"]:
            print(f"CUDA device: {dev['name']} ({dev['memory_mb']}MB) - unused")
    print(f"Kaggle: {env['is_kaggle']} runtime={env['kernel_run_type']}")
    print(f"Internet: {env['has_internet']}")

    cfg_path = REPO_ROOT / "kaggle" / "configs" / "kaggle_cpu.json"
    cfg = (
        FeatherV1Config.from_file(str(cfg_path))
        if cfg_path.is_file()
        else FeatherV1Config.auto()
    )
    model = FeatherV1Model(cfg)
    print(
        f"Model: dim={cfg.dim} hv={cfg.hypervector_dim} threads={cfg.threads} "
        f"vocab={cfg.vocab_size} ram_budget={cfg.ram_budget_gb}GB"
    )
    print(
        f"Kernel: {model.kernel['binding']}, expected "
        f"{model.kernel['expected_tok_per_sec']} on CPU"
    )


if __name__ == "__main__":
    main()
