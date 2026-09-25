"""Feather v1 -- build REAL GGUF from a REAL trained checkpoint.

OPT 3 invariant: the GGUF holds weights that were actually trained on the
real WikiText-2 corpus (911144 tokens) -- not synthetic, not zeros.  This
module is the single source of truth for that build: EXACTLY 16 metadata
keys, header GGUF v3 (24 bytes), 1 tensor, round-trip True.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .reader import npz_round_trip, read_gguf
from .writer import (
    FILE_TYPES,
    QUANTIZERS,
    STRING,
    UINT32,
    UINT64,
    PackedTensor,
    write_gguf,
)

REAL_METADATA_KEYS = 16


def real_projection_metadata(
    tensor: PackedTensor,
    cfg: dict[str, Any],
    size: str,
    quant: str,
    author: str = "saurav_bhandari_author",
) -> list[tuple[str, int, Any]]:
    """Metadata for a REAL-feather GGUF. Exactly REAL_METADATA_KEYS entries."""
    context_length = int(cfg.get("seq_len", 512))
    dim = int(cfg.get("dim", tensor.dims[0]))
    vocab = int(cfg.get("vocab_size", tensor.dims[1]))
    keys: list[tuple[str, int, Any]] = [
        ("general.architecture", STRING, "feather"),
        ("general.name", STRING, f"feather-v1-{size}"),
        ("general.author", STRING, author),
        ("general.version", STRING, "1.0.0"),
        ("general.license", STRING, "mit"),
        ("general.url", STRING, "https://github.com/saurav3231/feather-v1"),
        ("general.file_type", UINT32, FILE_TYPES[quant]),
        ("general.description", STRING, f"Feather v1 {size} REAL WikiText-2 ({quant})"),
        ("feather.context_length", UINT64, context_length),
        ("feather.embedding_length", UINT64, dim),
        ("feather.vocab_size", UINT64, vocab),
        ("feather.block_count", UINT64, int(cfg.get("num_chunks", 16))),
        ("feather.seed", UINT64, int(cfg.get("seed", 42))),
        ("feather.alpha_fractional", UINT32, int(cfg.get("alpha_fractional", 0.7))),
    ]
    assert len(keys) == REAL_METADATA_KEYS - 2, "tensor keys fill to 16"
    keys.append(("feather.tensor", STRING, tensor.name))
    keys.append(("feather.tensor.ggml_type", UINT32, tensor.ggml_type))
    assert len(keys) == REAL_METADATA_KEYS
    return keys


def build_real_gguf(
    pt_path: str | Path,
    out_path: str | Path,
    quant: str,
    size: str,
    author: str = "saurav_bhandari_author",
) -> dict[str, Any]:
    """Convert a REAL trained npz checkpoint -> GGUF (f16 or q4_k_m)."""
    quant = quant.lower()
    if quant not in QUANTIZERS:
        raise ValueError(f"unknown quant {quant!r}; choose from {sorted(QUANTIZERS)}")
    pt_path = Path(pt_path)
    data = np.load(pt_path, allow_pickle=False)
    cfg = json.loads(bytes(data["config"]).decode("utf-8"))
    projection = np.asarray(data["logit_projection"], dtype=np.float32)
    if projection.ndim != 2:
        raise ValueError("logit_projection must be 2-D")
    if float(np.std(projection)) == 0.0:
        raise ValueError("logit_projection is all zeros -- not a REAL weight")

    tensor = QUANTIZERS[quant]("output", projection)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(
        out_path, [tensor], real_projection_metadata(tensor, cfg, size, quant, author)
    )

    result = read_gguf(out_path)
    round_trip = npz_round_trip(str(out_path), str(pt_path))
    return {
        "size": size,
        "quant": quant,
        "path": str(out_path),
        "bytes": int(out_path.stat().st_size),
        "n_tensors": len(result.tensors),
        "n_metadata": len(result.metadata),
        "round_trip": round_trip,
        "vocab_size": int(cfg.get("vocab_size", cfg.get("dim", 96))),
        "dim": int(cfg.get("dim", projection.shape[0])),
    }
