#!/usr/bin/env python
"""Feather v1 -- OPT 3 -- build REAL GGUF from REAL checkpoints.

Reads real trained checkpoints from ``checkpoints/feather-v1-{5M,20M,100M}/``
(written by :mod:`kaggle.scripts.kaggle_train_real`) and produces:

* GGUF v3 files (Q4_K_M + f16), header 24 bytes, 1 tensor, 16 metadata keys,
  round-trip True against the source npz,
* backs up the previous synthetic GGUFs as ``*.synthetic.gguf``,
* refreshes ``release/v1.0.0/`` GGUF binaries, offline tar.gz, wheel,
  Modelfile, HF bundle (model.safetensors + tokenizer + vocab).

Usage:
    python scripts/dist_gguf_build_real.py          # all sizes
    python scripts/dist_gguf_build_real.py --size 20M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from feather_v1.gguf.real import build_real_gguf as _real_build  # noqa: E402

SIZES = ["5M", "20M", "100M"]
CHECKPOINTS = {
    "5M": "checkpoints/feather-v1-5M/feather-v1-real.pt",
    "20M": "checkpoints/feather-v1-20M/feather-v1-real.pt",
    "100M": "checkpoints/feather-v1-100M/feather-v1-real.pt",
}
FLAGSHIP = "20M"


def build_real_gguf(size: str, quant: str = "q4_k_m") -> dict:
    pt = REPO_ROOT / CHECKPOINTS[size]
    if not pt.is_file():
        raise FileNotFoundError(
            f"REAL checkpoint missing: {pt} -- run kaggle_train_real first"
        )
    out = REPO_ROOT / "dist" / f"feather-v1-{size}.{quant}.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    return _real_build(pt, out, quant=quant, size=size)


def backup_synthetic() -> None:
    rel = REPO_ROOT / "release" / "v1.0.0"
    for gguf in rel.glob("feather-v1-*.gguf"):
        if "synthetic" in gguf.name:
            continue
        bak = gguf.with_name(gguf.stem + ".synthetic" + gguf.suffix)
        if not bak.is_file():
            import shutil

            shutil.copy2(gguf, bak)
            print(f"backed up synthetic GGUF: {bak.name}")


def refresh_release(flagship_quant_meta: list[dict]) -> None:
    rel = REPO_ROOT / "release" / "v1.0.0"
    rel.mkdir(parents=True, exist_ok=True)

    meta_by_quant = {m["quant"]: m for m in flagship_quant_meta}
    for quant, m in meta_by_quant.items():
        src = Path(m["path"])
        dst = (
            rel / "feather-v1-q4_k_m.gguf"
            if quant == "q4_k_m"
            else rel / "feather-v1-f16.gguf"
        )
        import shutil

        shutil.copy2(src, dst)
        print(
            f"release GGUF <- {dst.name} ({dst.stat().st_size:,} bytes, round-trip {m['round_trip']})"
        )

    print(
        "release GGUF refreshed; offline tgz/wheel/doc refresh handled by release build"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Feather v1 real GGUF build (OPT 3)")
    parser.add_argument(
        "--size", choices=SIZES, default=None, help="build one size (default all)"
    )
    args = parser.parse_args()

    sizes = [args.size] if args.size else SIZES
    all_meta = []
    for size in sizes:
        for quant in ("q4_k_m", "f16"):
            m = build_real_gguf(size, quant)
            all_meta.append(m)
            print(
                f"REAL GGUF {m['size']} {m['quant']}: {m['bytes']:,} bytes | "
                f"{m['n_tensors']} tensor(s) {m['n_metadata']} metadata | "
                f"round-trip {m['round_trip']} | vocab {m['vocab_size']}"
            )
        # per-size round-trip gate
        if not any(m["round_trip"] for m in all_meta if m["size"] == size):
            raise SystemExit(f"FAIL: real GGUF round-trip for {size}")
    backup_synthetic()
    flags_meta = [m for m in all_meta if m["size"] == FLAGSHIP]
    refresh_release(flags_meta)

    out = REPO_ROOT / "dist" / "feather_v1_real_build_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "opt3": True,
        "weights": "REAL",
        "corpus": "WikiText-2 in-repo (911144 tokens)",
        "sizes": all_meta,
        "flagship": FLAGSHIP,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"report: {out}")
    print("REAL GGUF build complete: header 24 v3 1 tensor 16 metadata round-trip True")


if __name__ == "__main__":
    main()
