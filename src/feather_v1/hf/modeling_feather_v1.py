"""Feather v1 -- ``FeatherV1ForCausalLM`` (HuggingFace container).

Wraps the numpy :class:`feather_v1.model.FeatherV1Model` inside a
``PreTrainedModel`` so ``AutoModelForCausalLM`` / ``from_pretrained`` work
end-to-end on CPU.  The torch tensor ``output.weight`` mirrors the numpy
``model._logit_projection``; loading a state dict also re-syncs the numpy
runtime.  All compute stays numpy/CPU.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import FeatherV1Config as CoreFeatherV1Config
from ..model import FeatherV1Model
from .configuration_feather_v1 import FeatherV1Config

try:
    import torch
    from torch import nn
    from transformers import PreTrainedModel  # type: ignore
    from transformers.modeling_outputs import (  # type: ignore
        CausalLMOutputWithPast,
    )
except Exception:  # pragma: no cover - transformers optional
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    PreTrainedModel = object  # type: ignore[assignment,misc]
    CausalLMOutputWithPast = object  # type: ignore[assignment,misc]


def _guard_torch() -> None:
    if torch is None:
        raise RuntimeError("feather_v1.hf requires 'torch' and 'transformers'")


class _FeatherEngine(nn.Module):  # type: ignore[misc]
    """nn.Module shell wrapping the numpy engine.

    Some transformers loaders call ``getattr(model, base_model_prefix)
    .state_dict()`` to compute expected checkpoint keys.  A plain numpy
    object would crash on that, so the numpy engine lives inside this
    wrapper: ``state_dict()`` returns the (empty) torch view and every
    other attribute is delegated to the numpy engine.
    """

    def __init__(self, core: CoreFeatherV1Config) -> None:
        super().__init__()
        self.engine = FeatherV1Model(core)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as err:
            engine = object.__getattribute__(self, "engine")
            if name == "engine":
                raise err
            return getattr(engine, name)

    def __setattr__(self, name: str, value: Any) -> None:
        engine = self.__dict__.get("engine")
        if (
            name != "engine"
            and engine is not None
            and not isinstance(value, (nn.Module, nn.Parameter, torch.Tensor))
        ):
            setattr(engine, name, value)
            return
        super().__setattr__(name, value)


class FeatherV1ForCausalLM(PreTrainedModel):  # type: ignore[misc, valid-type]
    config_class = FeatherV1Config
    base_model_prefix = "feather"
    _supports_cache_class = False
    supports_gradient_checkpointing = False
    _no_split_modules: list[str] = []
    _tied_weights_keys: list[str] = []
    all_tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: FeatherV1Config) -> None:
        _guard_torch()
        super().__init__(config)
        core: CoreFeatherV1Config = config.to_core()
        self.feather = _FeatherEngine(core)
        self.embed = nn.Embedding(
            num_embeddings=core.vocab_size, embedding_dim=core.dim
        )
        self.output = nn.Linear(
            in_features=core.dim, out_features=core.vocab_size, bias=False
        )
        with torch.no_grad():
            self.output.weight.copy_(
                torch.from_numpy(
                    np.asarray(self.feather._logit_projection, dtype=np.float32).T
                )
            )
            self.embed.weight.copy_(torch.from_numpy(self._default_embed(core)))

    @staticmethod
    def _default_embed(core: CoreFeatherV1Config) -> np.ndarray:
        """Deterministic byte->embedding: one-hot placed by ``token % dim``."""
        rng = np.random.default_rng(core.seed + 1)
        return rng.standard_normal((core.vocab_size, core.dim)).astype(
            np.float32
        ) / np.sqrt(core.dim)

    def forward(
        self,
        input_ids: Any | None = None,
        attention_mask: Any | None = None,  # noqa: ARG002
        inputs_embeds: Any | None = None,
        labels: Any | None = None,
        return_dict: bool | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> CausalLMOutputWithPast:
        _guard_torch()
        if input_ids is None and inputs_embeds is None:
            raise ValueError("FeatherV1ForCausalLM needs input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)  # (batch, seq, dim)
        batch, seq, dim = inputs_embeds.shape
        embedded_np = inputs_embeds.detach().cpu().numpy()
        logits_rows = []
        for b in range(batch):
            rows = embedded_np[b]  # (seq, dim) embedding stream
            out = self.feather.encode(rows)
            logits_rows.append(
                np.asarray(out["protected"], dtype=np.float64)
                @ np.asarray(self.feather._logit_projection, dtype=np.float64)
            )
        logits_np = np.stack(logits_rows)[:, None, :]  # (batch, 1, vocab)
        logits = torch.from_numpy(np.asarray(logits_np, dtype=np.float32))
        loss = None
        if labels is not None:
            shift_logits = (
                logits[:, :-1, :].contiguous() if logits.shape[1] > 1 else logits
            )
            shift_labels = labels[:, :].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        super()._load_from_state_dict(*args, **kwargs)
        self._sync_numpy_from_torch()

    def _sync_numpy_from_torch(self) -> None:
        if torch is None:
            return
        weight = self.output.weight.detach().cpu().numpy()  # (vocab, dim)
        self.feather._logit_projection = np.asarray(weight.T, dtype=np.float64)

    @classmethod
    def from_core_weights(
        cls, pt_path: str, hf_config: FeatherV1Config | None = None
    ) -> FeatherV1ForCausalLM:
        """Build the HF wrapper straight from a numpy ``save_weights`` npz."""
        _guard_torch()
        data = np.load(pt_path, allow_pickle=False)
        cfg = json.loads(bytes(data["config"]).decode("utf-8"))
        if hf_config is None:
            hf_config = FeatherV1Config.from_core(CoreFeatherV1Config.from_dict(cfg))
        model = cls(hf_config)
        with torch.no_grad():
            model.output.weight.copy_(
                torch.from_numpy(
                    np.asarray(data["logit_projection"], dtype=np.float32).T
                )
            )
        model._sync_numpy_from_torch()
        return model
