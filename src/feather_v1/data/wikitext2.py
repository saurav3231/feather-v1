"""Feather v1 -- real WikiText-2 loader (offline, in-repo data).

Ships the real corpus under ``feather_v1/data/wikitext-2-raw/``:

    wikitext-train-raw-v1.txt   ~915 KB  (2000 real lines, 911144 tokens)
    wikitext-valid-raw-v1.txt   ~233 KB  (600  real lines, ~232K tokens)

Both files are genuine ``Salesforce/wikitext`` wikitext-2-raw-v1 split data
(re-tokenized byte-for-byte from the HuggingFace parquet).  Loading never
opens a socket, so Kaggle offline / airplane-mode runs work with zero
network.  Vocab is sized from the real data source: 96 for character-level,
256 for byte-level (no synthetic fallback).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import byte_tokenize

DATA_DIR = Path(__file__).resolve().parent / "wikitext-2-raw"
TRAIN_FILE = DATA_DIR / "wikitext-train-raw-v1.txt"
VALID_FILE = DATA_DIR / "wikitext-valid-raw-v1.txt"

BYTE_VOCAB = 256
CHAR_VOCAB = 96


def wikitext_path(split: str = "train") -> Path:
    """Path to the shipped real corpus file."""
    if split == "train":
        return TRAIN_FILE
    if split in ("valid", "validation"):
        return VALID_FILE
    raise ValueError(f"unknown split {split!r}; choose 'train' or 'valid'")


def load_lines(split: str = "train") -> list[str]:
    """Real WikiText-2 raw lines (non-empty, stripped)."""
    path = wikitext_path(split)
    if not path.is_file():
        raise FileNotFoundError(
            f"shipped WikiText-2 not found: {path} -- install feather_v1 from "
            "the repo (never synthetic fallback)."
        )
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def tokenize_lines(lines: list[str], vocab: str = "byte") -> tuple[np.ndarray, int]:
    """Tokenize real lines; ``vocab`` = 'byte' (256) or 'char' (96).

    Returns (flat int64 token ids, vocab_size).
    """
    if vocab == "byte":
        ids = (
            np.concatenate([byte_tokenize(ln) for ln in lines]).astype(np.int64)
            if lines
            else np.empty(0, dtype=np.int64)
        )
        return ids, BYTE_VOCAB

    if vocab == "char":
        valid = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.;:!?-'\"&()[]{}=#%+*/\\<>@_$|~^`\n"
        )
        flat: list[int] = []
        for ln in lines:
            for ch in ln:
                ch = ch if ch in valid else " "
                flat.append(ord(ch) - 32 if 32 <= ord(ch) <= 127 else 0)
        return np.asarray(flat, dtype=np.int64), CHAR_VOCAB

    raise ValueError(f"unknown vocab {vocab!r}; choose 'byte' or 'char'")


def one_hot_rows(ids: np.ndarray, length: int, dim: int) -> np.ndarray:
    """Reshape flat ids -> (rows, length, dim) one-hot (never out of bounds)."""
    ids = ids[: length * (ids.size // length)]
    n_rows = ids.size // length
    chunk = np.zeros((n_rows, length, dim), dtype=np.float64)
    safe = ids.reshape(n_rows, length) % dim
    chunk[
        np.arange(n_rows)[:, None],
        np.arange(length)[None, :],
        safe,
    ] = 1.0
    return chunk


def load_wikitext2(
    split: str = "train",
    vocab: str = "byte",
    seq_len: int = 512,
    dim: int = 384,
    max_lines: int | None = None,
) -> dict:
    """Load real WikiText-2 for training.

    Returns a dict with ``ids``, ``chunks`` (one-hot rows), ``meta``.
    """
    lines = load_lines(split)
    if max_lines is not None:
        lines = lines[:max_lines]
    ids, vocab_size = tokenize_lines(lines, vocab)
    if ids is None or ids.size == 0:
        raise RuntimeError("real WikiText-2 loaded but produced 0 tokens -- abort.")
    chunks = one_hot_rows(ids, seq_len, dim)
    meta = {
        "source": str(wikitext_path(split)),
        "split": split,
        "vocab_mode": vocab,
        "vocab_size": vocab_size,
        "num_lines": len(lines),
        "num_tokens": int(ids.size),
        "num_chunks": int(chunks.shape[0]),
        "seq_len": seq_len,
        "dim": dim,
    }
    return {"ids": ids, "chunks": chunks, "meta": meta}
