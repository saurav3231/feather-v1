"""Kaggle -- offline inference on bare CPU.

Loads the packaged model built by ``kaggle_dataset.py`` (plain ``.npz`` or the
``tar.gz``), tokenizes a prompt with the byte-level tokenizer (single source:
``feather_v1.utils``), and prints what the six-component pipeline produces.
No network, no GPU, no torch.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import numpy as np

from feather_v1 import FeatherV1Model
from feather_v1.utils import byte_decode, byte_tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]

WEIGHT_CANDIDATES = (
    REPO_ROOT / "models" / "kaggle" / "feather-v1-kaggle.pt",
    Path("/kaggle/input/feather-v1-model/feather-v1-kaggle.pt"),
    Path("/kaggle/input/feather-v1-model/feather-v1-kaggle.tar.gz"),
)


def find_weights() -> Path | None:
    for cand in WEIGHT_CANDIDATES:
        if cand.is_file():
            return cand
    return None


def load_model(weights: Path) -> FeatherV1Model:
    if weights.suffix == ".tar.gz":
        with tarfile.open(weights, "r:gz") as tf:
            members = tf.getnames()
            pt = next((m for m in members if m.endswith(".pt")), None)
            if pt is None:
                raise SystemExit(f"no packaged .pt inside {weights}")
            dest = Path(tempfile.mkdtemp())
            tf.extract(pt, dest)
            return FeatherV1Model.from_weights(str(dest / pt))
    return FeatherV1Model.from_weights(str(weights))


def prompt_rows(text: str, dim: int) -> np.ndarray:
    ids = byte_tokenize(text)
    ids = ids[: dim // 2]
    prompt = np.zeros((ids.size, dim))
    for i, tok in enumerate(ids):
        prompt[i, int(tok) % dim] = 1.0
    return prompt


def main() -> None:
    weights = find_weights()
    if weights is None:
        raise SystemExit(
            "no packaged weights - run kaggle/scripts/kaggle_dataset.py first"
        )
    model = load_model(weights)
    dim = model.config.dim

    prompt_text = "Feather v1 runs CPU-only on Kaggle."
    prompt = prompt_rows(prompt_text, dim)
    seq = model.generate(prompt, steps=8)
    token = int(np.argmax(seq[-1]))
    word = byte_decode(np.asarray([token % 256]))

    print(f"Weights: {weights}")
    print(f"Kernel: {model.kernel['binding']} ({model.kernel['hypervector_dim']}-D)")
    print(f"Expected: {model.kernel['expected_tok_per_sec']} (CPU-only)")
    print(f"Prompt ({prompt.shape[0]} bytes as one-hot rows): {prompt_text!r}")
    print(f"Generated token: id {token} -> {word!r} (byte-level decoder)")


if __name__ == "__main__":
    main()
