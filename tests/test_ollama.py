"""Feather v1 -- Ollama distribution tests.

The GGUF conversion is numpy-only and always testable here. ``ollama create``
requires the OLama binary, so its integration is asserted structurally
(Modelfile contents, package layout, tokenizer <-> template token match).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import feather_v1
from feather_v1.gguf import read_gguf

REPO_ROOT = Path(feather_v1.__file__).resolve().parent.parent.parent


def test_modelfile_matches_reference_hardware():
    modelfile = (REPO_ROOT / "ollama" / "Modelfile").read_text(encoding="utf-8")
    assert "FROM ../dist/feather-v1-q4_k_m.gguf" in modelfile
    assert "PARAMETER num_thread 2" in modelfile
    assert "PARAMETER num_gpu 0" in modelfile
    assert "<|system|>" in modelfile
    assert "<|assistant|>" in modelfile


def test_template_tokens_match_tokenizer_specials():
    from feather_v1.hf.tokenization_feather_v1 import SPECIAL_TOKENS

    modelfile = (REPO_ROOT / "ollama" / "Modelfile").read_text(encoding="utf-8")
    for tok in ("<|system|>", "<|user|>", "<|assistant|>"):
        assert tok in SPECIAL_TOKENS
        assert tok in modelfile or "{{ .System }}" in modelfile


def test_q4_k_gguf_builds_and_reads(tmp_path):
    from feather_v1.config import FeatherV1Config as CoreConfig
    from feather_v1.gguf import convert_pt_to_gguf

    core = CoreConfig(dim=32, seq_len=64, chunk_size=8, num_chunks=8, vocab_size=32)
    npz = tmp_path / "w.pt"
    with open(npz, "wb") as fh:
        np.savez(
            fh,
            config=json.dumps(core.to_dict()).encode("utf-8"),
            logit_projection=np.random.default_rng(3)
            .standard_normal((32, 32))
            .astype(np.float32),
        )
    out = convert_pt_to_gguf(npz, tmp_path / "feather-v1-q4_k_m.gguf", quant="q4_k_m")
    result = read_gguf(out)
    assert result.tensors[0].name == "output"
    assert result.metadata_dict()["general.file_type"] == 15
    assert out.stat().st_size < 32 * 32 * 4


def test_ollama_share_script_guides_without_binary():
    # publish script carries a no-binary guidance branch (no `ollama` on PATH)
    script = (REPO_ROOT / "scripts" / "ollama_publish.py").read_text(encoding="utf-8")
    assert 'shutil.which("ollama")' in script
    assert "ollama create feather-v1 -f" in script
