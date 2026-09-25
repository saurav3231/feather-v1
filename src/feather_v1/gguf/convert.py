"""Feather v1 -- convert a numpy weight bundle to GGUF v3.

Input is a ``feather-v1-kaggle.pt`` npz with ``config`` (JSON str) and
``logit_projection`` (dim x vocab).  Quantization chooses per-tensor
``PackedTensor``; ``general.file_type`` metadata says what the GGUF file
holds.  Output round-trips exactly through :func:`feather_v1.gguf.npz_round_trip`
for lossless quants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .writer import (
    FILE_TYPES,
    QUANTIZERS,
    STRING,
    UINT32,
    UINT64,
    PackedTensor,
    write_gguf,
)

DEFAULT_QUANT = "f16"


def _tensor_metadata(tensor: PackedTensor) -> list[tuple[str, int, Any]]:
    return [
        ("feather.tensor", STRING, tensor.name),
        ("feather.tensor.ggml_type", UINT32, tensor.ggml_type),
        ("feather.tensor.n_elements", UINT64, tensor.n_elements),
    ]


def convert_pt_to_gguf(
    pt_path: str | Path,
    out_path: str | Path,
    quant: str = DEFAULT_QUANT,
    name: str = "output",
    author: str = "saurav_bhandari_author",
    description: str = "",
) -> Path:
    """Convert npz weights -> quantized GGUF v3 file."""
    quant = quant.lower()
    if quant not in QUANTIZERS:
        raise ValueError(f"unknown quant {quant!r}; choose from {sorted(QUANTIZERS)}")
    pt_path = Path(pt_path)
    data = np.load(pt_path, allow_pickle=False)
    cfg = json.loads(bytes(data["config"]).decode("utf-8"))
    projection = np.asarray(data["logit_projection"], dtype=np.float32)
    if projection.ndim != 2:
        raise ValueError("logit_projection must be 2-D")

    pack = QUANTIZERS[quant]
    tensor = pack(name, projection)
    context_length = int(cfg.get("seq_len", 512))
    dim = int(cfg.get("dim", projection.shape[0]))
    metadata: list[tuple[str, int, Any]] = [
        ("general.architecture", STRING, "feather"),
        ("general.name", STRING, "feather-v1"),
        ("general.author", STRING, author),
        ("general.version", STRING, "1.0.0"),
        ("general.license", STRING, "mit"),
        ("general.url", STRING, "https://github.com/saurav3231/feather-v1"),
        ("general.file_type", UINT32, FILE_TYPES[quant]),
        ("feather.context_length", UINT64, context_length),
        ("feather.embedding_length", UINT64, dim),
        ("feather.vocab_size", UINT64, int(cfg.get("vocab_size", 4096))),
        ("feather.block_count", UINT64, int(cfg.get("num_chunks", 16))),
        ("feather.seed", UINT64, int(cfg.get("seed", 42))),
        ("feather.alpha_fractional", UINT32, int(cfg.get("alpha_fractional", 0.7))),
    ]
    description = (
        description
        or f"Feather v1 ({quant}), {projection.shape[0]}x{projection.shape[1]} projection"
    )
    metadata.append(("general.description", STRING, description))
    metadata.extend(_tensor_metadata(tensor))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(out_path, [tensor], metadata)
    return out_path
