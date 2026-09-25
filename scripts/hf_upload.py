#!/usr/bin/env python
"""Build + upload the Feather v1 HuggingFace model.

Usage:
    HF_TOKEN=hf_... python scripts/hf_upload.py            # build + publish
    python scripts/hf_upload.py --build-only               # build bundle only

Requires ``transformers`` + ``torch`` + ``huggingface_hub``.  With no
``HF_TOKEN`` set the bundle is still built locally (for inspection / a
manual ``huggingface-cli upload``) and the publish step prints guidance.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def build_bundle(out_dir: Path, pt_path: Path) -> Path:
    from feather_v1.hf import convert_core_to_hf

    if not pt_path.is_file():
        raise SystemExit(f"model weights not found: {pt_path}")
    print(f"converting {pt_path} -> {out_dir}")
    return convert_core_to_hf(pt_path, out_dir)


def publish(bundle_path: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN not set. Bundle built; publish yourself with:\n"
            f"  huggingface-cli upload --repo-type model {repo_id} {bundle_path} .\n"
        )
    from huggingface_hub import HfApi  # type: ignore

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(bundle_path),
        repo_type="model",
        commit_message="feather-v1 distribution phase: HF bundle",
    )
    print(f"live at https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feather v1 HF upload")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--repo", default="saurav3231/feather-v1")
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "dist" / "hf"), help="bundle output dir"
    )
    parser.add_argument(
        "--pt",
        default=str(REPO_ROOT / "models" / "kaggle" / "feather-v1-kaggle.pt"),
        help="source numpy weights",
    )
    args = parser.parse_args()
    bundle = build_bundle(Path(args.out), Path(args.pt))
    if not args.build_only:
        publish(bundle, args.repo)


if __name__ == "__main__":
    main()
