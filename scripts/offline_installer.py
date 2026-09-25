#!/usr/bin/env python
"""Feather v1 -- fully offline distribution bundle ("Pokhara bundle").

Packs everything needed to run on an internet-blocked machine:

    feather-v1-offline-v1.0.0.tar.gz
    ├── install.sh                 # pip install --no-index --find-links=wheels
    ├── wheels/                    # feather_v1 wheel + deps (numpy, py-cpuinfo)
    ├── feather-v1-q4_k_m.gguf     # quantized weights (offline model)
    ├── configs/i5_3337U.json      # reference CPU tuning
    ├── configs/kaggle_cpu.json
    └── README.txt                 # air-gapped quick start

The wheel is built with `pip wheel` (which also fetches numpy/py-cpuinfo once
on the *authoring* machine); on the offline machine install.sh needs only the
tarball plus an existing Python 3.8+ with pip.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

from feather_v1.gguf import convert_pt_to_gguf

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST = REPO_ROOT / "dist"
VERSION = "1.0.0"
NAME = f"feather-v1-offline-v{VERSION}"


def _build_wheel(wheels_dir: Path) -> Path:
    print("building wheels (authoring machine only)...")
    wheels_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "-w", str(wheels_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    wheels = sorted(wheels_dir.glob("feather_v1-*.whl"))
    if not wheels:
        raise SystemExit("wheel build produced nothing")
    return wheels[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline bundle builder")
    parser.add_argument("--quant", default="q4_k_m", help="GGUF quant in the bundle")
    parser.add_argument("--no-wheels", action="store_true", help="skip pip wheel build")
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "models" / "kaggle" / "feather-v1-kaggle.pt"),
        help="source numpy weights npz",
    )
    args = parser.parse_args()

    pt = Path(args.weights)

    stage = DIST / NAME
    wheels = stage / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    (stage / "configs").mkdir(parents=True, exist_ok=True)
    if not args.no_wheels:
        _build_wheel(wheels)

    gguf = stage / "feather-v1-q4_k_m.gguf"
    if not gguf.is_file():
        convert_pt_to_gguf(pt, gguf, quant=args.quant)
    model_npz = stage / "feather-v1-kaggle.pt"
    if not model_npz.is_file():
        import shutil

        shutil.copy2(pt, model_npz)

    for cfg in (REPO_ROOT / "configs").glob("*.json"):
        import shutil

        shutil.copy2(cfg, stage / "configs" / cfg.name)

    install_sh = stage / "install.sh"
    install_sh.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "# Feather v1 offline install (no network required)",
                "set -e",
                'python -m pip install --no-index --find-links="$(dirname "$0")/wheels" '
                '"$(dirname "$0")/wheels"/feather_v1-*.whl',
                'echo "ok -- try: python -m feather_v1.demo --gguf feather-v1-q4_k_m.gguf"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    install_sh.chmod(0o755)

    readme = stage / "README.txt"
    readme.write_text(
        "Feather v1 offline bundle (v1.0.0)\n"
        "--------------------------------\n"
        "1. python install.sh\n"
        "2. Get a model:\n"
        "      model = FeatherV1Model.from_weights('feather-v1-kaggle.pt')\n"
        "      text  = model.generate(prompt_tokens, steps=16)\n"
        "   (the .gguf is the Ollama/llama.cpp portable format; the .pt npz is\n"
        "    the numpy engine weight file.)\n"
        "3. Works in airplane mode: feather_v1 opens no sockets.\n",
        encoding="utf-8",
    )

    tarball = DIST / f"{NAME}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(stage, arcname=NAME)
    print(f"bundle: {tarball} ({tarball.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
