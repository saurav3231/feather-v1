"""Feather v1 -- HuggingFace transformers integration (CPU-only).

Exposes ``FeatherV1Config`` (PretrainedConfig), ``FeatherV1Tokenizer`` and
``FeatherV1ForCausalLM`` and registers them with ``transformers`` Auto API
(model_type ``feather``) so that::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("saurav3231/feather-v1")

works on CPU without CUDA.  Everything stays numpy/CPU inside; torch is only
the HF container/reference-counting layer.  ``transformers`` and ``torch`` are
optional imports so the offline numpy runtime never requires them.
"""

from __future__ import annotations

try:
    from transformers import (  # type: ignore
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
except Exception:  # pragma: no cover - transformers optional
    AutoConfig = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from .configuration_feather_v1 import FeatherV1Config
from .convert import convert_core_to_hf
from .modeling_feather_v1 import FeatherV1ForCausalLM
from .tokenization_feather_v1 import FeatherV1Tokenizer

MODEL_TYPE = "feather"

if AutoConfig is not None and not hasattr(AutoConfig, "_feather_registered"):
    try:
        AutoConfig.register(MODEL_TYPE, FeatherV1Config)
    except Exception:  # pragma: no cover - transformers API drift
        pass
    AutoConfig._feather_registered = True


if AutoModelForCausalLM is not None and not hasattr(  # pragma: no cover
    AutoModelForCausalLM, "_feather_registered"
):
    try:
        AutoModelForCausalLM.register(MODEL_TYPE, FeatherV1ForCausalLM)
    except Exception:  # pragma: no cover - transformers API drift
        pass
    AutoModelForCausalLM._feather_registered = True


if AutoTokenizer is not None and not hasattr(  # pragma: no cover
    AutoTokenizer, "_feather_registered"
):
    try:
        AutoTokenizer.register(MODEL_TYPE, FeatherV1Tokenizer)
    except Exception:  # pragma: no cover - transformers API drift
        pass
    AutoTokenizer._feather_registered = True

__all__ = [
    "MODEL_TYPE",
    "FeatherV1Config",
    "FeatherV1Tokenizer",
    "FeatherV1ForCausalLM",
    "convert_core_to_hf",
]
