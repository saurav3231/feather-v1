"""Kaggle -- package Feather v1 for offline use (dataset mechanism).

Builds ``models/kaggle/`` containing:
  - ``feather-v1-kaggle.pt``   (npz: config + projection, loadable anywhere)
  - ``feather-v1-kaggle.gguf`` (spec-compliant GGUF v3: header + metadata +
                                F32 tensor infos + aligned tensor data)
  - ``tokenizer.json``         (byte-level tokenizer descriptor)
  - ``config.json``            (full Feather v1 config)
  - ``README.md``              (runbook for the packaged artifact)
  - ``dataset-metadata.json``  (Kaggle CLI dataset descriptor)
  - ``feather-v1-kaggle.tar.gz`` (the five files above, one Kaggle input)

Then ``kaggle_inference.py`` (or a notebook behind /kaggle/input) loads the
``.pt``; GGUF is emitted for third-party executors. No torch, no internet.
"""

from __future__ import annotations

import json
import os
import struct
import tarfile
from pathlib import Path

import numpy as np

from feather_v1 import FeatherV1Config, FeatherV1Model

REPO_ROOT = Path(__file__).resolve().parents[2]

# GGUF value types (gguf.md v3)
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _F32, _BOOL = range(8)
_STRING, _ARRAY, _UINT64, _INT64, _F64 = 8, 9, 10, 11, 12
_ALIGN = 32


def _w_string(fh, text: str) -> None:
    raw = text.encode("utf-8")
    fh.write(struct.pack("<Q", len(raw)))
    fh.write(raw)


def _w_value(fh, vtype: int, value) -> None:
    if vtype == _STRING:
        _w_string(fh, str(value))
    elif vtype == _UINT32:
        fh.write(struct.pack("<I", int(value)))
    elif vtype == _INT32:
        fh.write(struct.pack("<i", int(value)))
    elif vtype == _F32:
        fh.write(struct.pack("<f", float(value)))
    elif vtype == _F64:
        fh.write(struct.pack("<d", float(value)))
    elif vtype == _UINT64:
        fh.write(struct.pack("<Q", int(value)))
    elif vtype == _BOOL:
        fh.write(struct.pack("<B", 1 if value else 0))
    elif vtype == _ARRAY:
        elem, items = value
        fh.write(struct.pack("<I", int(elem)))
        fh.write(struct.pack("<Q", len(items)))
        for item in items:
            _w_value(fh, int(elem), item)


def write_gguf(
    path: Path, tensors: dict[str, np.ndarray], metadata: list[tuple[str, int, object]]
) -> None:
    """Minimal but spec-correct GGUF v3 writer (little-endian)."""
    blocks: list[tuple[str, np.ndarray, int]] = []
    off = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        off = (off + _ALIGN - 1) // _ALIGN * _ALIGN
        blocks.append((name, arr, off))
        off += arr.nbytes

    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))
        fh.write(struct.pack("<Q", len(tensors)))
        fh.write(struct.pack("<Q", len(metadata)))
        for key, vtype, value in metadata:
            _w_string(fh, key)
            fh.write(struct.pack("<I", vtype))
            _w_value(fh, vtype, value)

        for name, arr, offset in blocks:
            _w_string(fh, name)
            fh.write(struct.pack("<I", len(arr.shape)))
            for dim in arr.shape[::-1]:
                fh.write(struct.pack("<Q", dim))
            fh.write(struct.pack("<I", 0))  # GGML_TYPE_F32
            fh.write(struct.pack("<Q", offset))

        pad = (_ALIGN - fh.tell() % _ALIGN) % _ALIGN
        fh.write(b"\x00" * pad)

        for _, arr, _ in blocks:
            padding = (_ALIGN - 0) % _ALIGN
            if padding:
                fh.write(b"\x00" * padding)
            fh.write(arr.tobytes())


def inspect_gguf(path: Path) -> None:
    """Read back the GGUF header/infos we wrote (self-validation)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    pos = 0
    assert raw[:4] == b"GGUF"
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", raw, 4)
    pos = 4 + 4 + 8 + 8
    keys: list[str] = []
    for _ in range(n_kv):
        (klen,) = struct.unpack_from("<Q", raw, pos)
        pos += 8
        keys.append(raw[pos : pos + klen].decode("utf-8"))
        pos += klen
        (vtype,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        pos = _skip_value(raw, pos, vtype)
    tensor_names: list[str] = []
    for _ in range(n_tensors):
        (nlen,) = struct.unpack_from("<Q", raw, pos)
        pos += 8
        tensor_names.append(raw[pos : pos + nlen].decode("utf-8"))
        pos += nlen
        (ndims,) = struct.unpack_from("<I", raw, pos)
        pos += 4 + ndims * 8 + 4 + 8
    print(f"  GGUF ok: version {version}, {n_tensors} tensors, {n_kv} metadata keys")
    print(f"  metadata: {', '.join(keys)}")
    print(f"  tensors: {', '.join(tensor_names)}")


def _skip_value(raw: bytes, pos: int, vtype: int) -> int:
    if vtype in (_UINT8, _INT8, _BOOL):
        return pos + 1
    if vtype in (_UINT16, _INT16):
        return pos + 2
    if vtype in (_UINT32, _INT32, _F32):
        return pos + 4
    if vtype in (_UINT64, _INT64, _F64):
        return pos + 8
    if vtype == _STRING:
        (n,) = struct.unpack_from("<Q", raw, pos)
        return pos + 8 + n
    if vtype == _ARRAY:
        (elem,) = struct.unpack_from("<I", raw, pos)
        (count,) = struct.unpack_from("<Q", raw, pos + 4)
        pos += 12
        for _ in range(count):
            pos = _skip_value(raw, pos, elem)
        return pos
    raise ValueError(f"unsupported gguf value type {vtype}")


def build_tokenizer_json() -> dict:
    vocab = {chr(i): i for i in range(256)}
    return {
        "version": "1.0.0",
        "model_type": "byte-level-bpe",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
            "byte_fallback": False,
            "ignore_merges": True,
        },
        "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}]},
        "pre_tokenizer": {"type": "ByteLevel"},
        "post_processor": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "unk_token": None,
    }


def main() -> None:
    out_dir = REPO_ROOT / "models" / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = REPO_ROOT / "kaggle" / "configs" / "kaggle_cpu.json"
    cfg = (
        FeatherV1Config.from_file(str(cfg_path))
        if cfg_path.is_file()
        else FeatherV1Config(dim=384, vocab_size=4096)
    )
    model = FeatherV1Model(cfg)

    pt_path = out_dir / "feather-v1-kaggle.pt"
    model.save_weights(str(pt_path))

    tensors = {"output.weight": model._logit_projection}
    tokens = [chr(i) for i in range(256)]
    metadata = [
        ("general.architecture", _STRING, "feather"),
        ("general.alignment", _UINT32, _ALIGN),
        ("general.name", _STRING, "feather-v1-kaggle"),
        ("general.author", _STRING, "saurav3231"),
        ("general.version", _STRING, "1.0.0"),
        ("general.license", _STRING, "MIT"),
        ("general.url", _STRING, "https://github.com/saurav3231/feather-v1"),
        ("general.file_type", _UINT32, 0),
        ("general.tags", _ARRAY, (_STRING, ["llm", "cpu", "fractional-calculus"])),
        ("feather.context_length", _UINT64, cfg.seq_len),
        ("feather.embedding_length", _UINT64, cfg.dim),
        ("feather.hypervector_dim", _UINT64, cfg.hypervector_dim),
        ("feather.expert_count", _UINT32, cfg.n_experts),
        ("feather.expert_used_count", _UINT32, cfg.moe_top_k),
        ("tokenizer.ggml.model", _STRING, "gpt2"),
        ("tokenizer.ggml.tokens", _ARRAY, (_STRING, tokens)),
    ]
    gguf_path = out_dir / "feather-v1-kaggle.gguf"
    write_gguf(gguf_path, tensors, metadata)

    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(build_tokenizer_json(), indent=2), encoding="utf-8"
    )

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "Feather v1 Kaggle package\n\n"
        "- feather-v1-kaggle.pt: numpy weights (FeatherV1Model.from_weights)\n"
        "- feather-v1-kaggle.gguf: GGUF v3 (logit projection, F32)\n"
        "- tokenizer.json: byte-level tokenizer descriptor\n"
        "- config.json: FeatherV1Config as JSON\n",
        encoding="utf-8",
    )

    dataset_metadata_path = out_dir / "dataset-metadata.json"
    username = os.environ.get("KAGGLE_USERNAME", "YOUR_USERNAME")
    dataset_metadata_path.write_text(
        json.dumps(
            {
                "id": f"{username}/feather-v1-model",
                "title": "feather-v1-model",
                "subtitle": "Feather v1 offline CPU runtime archive",
                "license": "Apache-2.0",
                "isPrivate": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tar_path = out_dir / "feather-v1-kaggle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name in (pt_path, gguf_path, tokenizer_path, config_path, readme_path):
            tf.add(name, arcname=f"feather-v1-kaggle/{name.name}")

    print(f"Wrote {len(list(out_dir.iterdir()))} artifacts to {out_dir}")
    for child in sorted(out_dir.iterdir()):
        print(f"  {child.stat().st_size:>9,} B  {child.name}")
    inspect_gguf(gguf_path)

    reloaded = FeatherV1Model.from_weights(str(pt_path))
    same = np.allclose(reloaded._logit_projection, model._logit_projection)
    print(
        f"npz round-trip: {same} (dim {reloaded.config.dim}, "
        f"vocab {reloaded.config.vocab_size})"
    )
    print("Archive ready for /kaggle/input upload.")


if __name__ == "__main__":
    main()
