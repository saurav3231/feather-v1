#!/usr/bin/env python
"""Build the Ollama GGUF and (if the `ollama` CLI exists) publish it.

    python scripts/ollama_publish.py                 # convert + ollama create
    python scripts/ollama_publish.py --skip-quant    # just ollama create

The conversion itself is dependency-light (numpy only). `ollama create`
needs the Ollama binary; when it is missing we print drop-in shell commands.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    from feather_v1.gguf import convert_pt_to_gguf

    parser = argparse.ArgumentParser(description="Feather v1 -> Ollama")
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument(
        "--quant",
        default="q4_k_m",
        help="GGUF quant to build (default q4_k_m)",
    )
    args = parser.parse_args()
    gguf = REPO_ROOT / "dist" / f"feather-v1-{args.quant}.gguf"
    if not args.skip_quant:
        pt = REPO_ROOT / "models" / "kaggle" / "feather-v1-kaggle.pt"
        if not pt.is_file():
            raise SystemExit(f"weights not found: {pt}")
        gguf = convert_pt_to_gguf(pt, gguf, quant=args.quant)
        print(f"built {gguf} ({gguf.stat().st_size:,} bytes)")
    modelfile = REPO_ROOT / "ollama" / "Modelfile"
    if shutil.which("ollama") is None:
        print(
            f"no `ollama` binary on PATH; run manually:\n  ollama create feather-v1 -f {modelfile}"
        )
        return
    subprocess.run(["ollama", "create", "feather-v1", "-f", str(modelfile)], check=True)
    print('done:  ollama run feather-v1 "namaste, xasan"')


if __name__ == "__main__":
    main()
