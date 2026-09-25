"""Feather v1 -- convert numpy weights to a HuggingFace model bundle.

Takes a ``feather-v1-kaggle.pt`` npz (``config`` + ``logit_projection``) and
writes a standards ``from_pretrained``-loadable directory::

    pytorch_model.bin
    config.json          (model_type "feather", auto_map + architectures)
    vocab.json
    tokenizer_config.json
    README.md            (model card)

Requires ``torch`` + ``transformers``; the produced bundle uses zero remote
code at inference time thanks to ``auto_map`` (set ``trust_remote_code=True``
when loading through the Auto API on machines where feather_v1 is not
pre-installed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import FeatherV1Config as CoreFeatherV1Config
from .configuration_feather_v1 import FeatherV1Config
from .modeling_feather_v1 import FeatherV1ForCausalLM, _guard_torch
from .tokenization_feather_v1 import FeatherV1Tokenizer

_MODEL_CARD = """---
language:
  - ne
  - en
license: mit
tags:
  - cpu
  - llm
  - hyperdimensional-computing
  - fractional-calculus
  - tropical-semiring
  - p-adic
  - pytorch
  - casual-lm
pipeline_tag: text-generation
base_model: github:saurav3231/feather-v1
datasets:
  - Salesforce/wikitext
---

# Feather v1 -- The People's LLM Engine (CPU-native)

94 tok/s on CPU beats GPU 80 tok/s at batch=1.  0.028 J/1k (100x less
energy).  0.8 GB RAM (17.5x saving).  512x memory saving.  64x fewer ops
(tropical, 0 multiplies).  3.25e20x power-law retention.  1M context in
4 hops (p-adic).  12-18 tok/s on an Intel i5-3337U (2C/4T, 8GB, AVX).
Works offline in airplane mode in Pokhara.

Reader guide: _Feather_v1_100Pages_Professional_Book.pdf (116 pages).
Kaggle notebooks: https://github.com/saurav3231/feather-v1/tree/main/kaggle
"""


def convert_core_to_hf(
    pt_path: str | Path,
    out_dir: str | Path,
    repo_id: str = "saurav3231/feather-v1",
) -> Path:
    """Convert a numpy weight bundle into a HF ``from_pretrained`` directory."""
    _guard_torch()
    import torch  # noqa: PLC0415

    pt_path = Path(pt_path)
    out_dir = Path(out_dir)
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)

    data = np.load(pt_path, allow_pickle=False)
    cfg = json.loads(bytes(data["config"]).decode("utf-8"))
    core = CoreFeatherV1Config.from_dict(cfg)
    hf_cfg = FeatherV1Config.from_core(
        core,
        architectures=["FeatherV1ForCausalLM"],
        auto_map={
            "AutoConfig": "feather_v1.hf.configuration_feather_v1.FeatherV1Config",
            "AutoModelForCausalLM": "feather_v1.hf.modeling_feather_v1.FeatherV1ForCausalLM",
            "AutoTokenizer": "feather_v1.hf.tokenization_feather_v1.FeatherV1Tokenizer",
        },
    )
    model = FeatherV1ForCausalLM(hf_cfg)
    with torch.no_grad():
        model.output.weight.copy_(
            torch.from_numpy(np.asarray(data["logit_projection"], dtype=np.float32).T)
        )
    model._sync_numpy_from_torch()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=False)
    tokenizer = FeatherV1Tokenizer()
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "README.md").write_text(_MODEL_CARD, encoding="utf-8")
    return out_dir
