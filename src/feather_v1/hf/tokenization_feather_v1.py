"""Feather v1 -- byte-level branching tokenizer (Nepali + English).

Reuses :func:`feather_v1.utils.byte_tokenize` (single source of truth) mapped
through a reversible ``byte:NN`` vocab, so every token is a single UTF-8 byte
and everything encodes/decodes losslessly without SentencePiece or the
``tokenizers`` library.  ``transformers`` is optional.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from transformers import PreTrainedTokenizer  # type: ignore
except Exception:  # pragma: no cover - transformers optional
    PreTrainedTokenizer = object  # type: ignore[assignment,misc]

from ..utils import byte_tokenize

BYTE_PREFIX = "byte:"

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


def _byte_token(token: str) -> int:
    assert token.startswith(BYTE_PREFIX), f"unexpected token {token!r}"
    return int(token[len(BYTE_PREFIX) :])


def build_vocab() -> dict[str, int]:
    vocab = {f"{BYTE_PREFIX}{b}": b for b in range(256)}
    for i, tok in enumerate(SPECIAL_TOKENS, start=256):
        vocab[tok] = i
    return vocab


def _build_itos(vocab: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in vocab.items()}


class FeatherV1Tokenizer(PreTrainedTokenizer):  # type: ignore[misc, valid-type]
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        vocab_file: str | None = None,
        pad_token: str = "<pad>",
        eos_token: str = "<|endoftext|>",
        **kwargs: Any,
    ) -> None:
        if vocab is None and vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as fh:
                vocab = json.load(fh)
        self.vocab: dict[str, int] = dict(vocab) if vocab else build_vocab()
        self.itos: dict[int, str] = _build_itos(self.vocab)
        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        return [f"{BYTE_PREFIX}{int(b)}" for b in byte_tokenize(text)]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["<pad>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self.itos.get(index, "<pad>")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raw = bytes(_byte_token(t) for t in tokens if t.startswith(BYTE_PREFIX))
        return raw.decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        from pathlib import Path as _Path

        out = _Path(save_directory) / f"{filename_prefix or ''}vocab.json"
        out.write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return (str(out),)

    @classmethod
    def from_vocab_file(cls, vocab_file: str, **kwargs: Any) -> FeatherV1Tokenizer:
        with open(vocab_file, encoding="utf-8") as fh:
            vocab = json.load(fh)
        return cls(vocab=vocab, **kwargs)
