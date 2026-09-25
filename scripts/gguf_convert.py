#!/usr/bin/env python
"""Convert Feather v1 numpy weights to a quantized GGUF v3 file.

    python scripts/gguf_convert.py --input models/kaggle/feather-v1-kaggle.pt \
        --output dist/feather-v1-f16.gguf --quant f16
    python scripts/gguf_convert.py --quant q8_0     # llama.cpp-verified
    python scripts/gguf_convert.py --quant q4_k_m   # feather-documented 2-bit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    from feather_v1.gguf import QUANTIZERS, convert_pt_to_gguf

    parser = argparse.ArgumentParser(description="Feather v1 -> GGUF v3")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "models" / "kaggle" / "feather-v1-kaggle.pt"),
        help="source numpy weights npz",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "dist" / "feather-v1.gguf"),
        help="output .gguf path",
    )
    parser.add_argument(
        "--quant",
        choices=sorted(QUANTIZERS),
        default="f16",
        help="quantization: llama.cpp-verified f16/q8_0, or q4_k_m",
    )
    args = parser.parse_args()
    path = convert_pt_to_gguf(args.input, args.output, quant=args.quant)
    print(f"wrote {path} ({path.stat().st_size:,} bytes, quant={args.quant})")


if __name__ == "__main__":
    main()
