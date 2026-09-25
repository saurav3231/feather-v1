"""Feather v1 -- HuggingFace integration tests (HF export + load round-trip).

These only run when ``torch`` + ``transformers`` are installed (the ``hf``
extra); the offline numpy runtime never requires them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def hf_config():
    from feather_v1.hf import FeatherV1Config

    core = make_core_config()
    return FeatherV1Config.from_core(core)


def make_core_config():
    from tests.conftest import make_config

    return make_config(dim=32, seq_len=64, chunk_size=8, num_chunks=8, vocab_size=64)


def test_hf_config_roundtrips_core_fields(hf_config):
    core = hf_config.to_core()
    assert core.dim == 32
    assert core.vocab_size == 64
    assert hf_config.model_type == "feather"
    assert hf_config.to_dict()["model_type"] == "feather"


def test_tokenizer_reversible_bytes_and_specials():
    from feather_v1.hf import FeatherV1Tokenizer

    tok = FeatherV1Tokenizer()
    text = "Feather v1 नेपाली + English"
    ids = tok.encode(text, add_special_tokens=False)
    assert all(0 <= i < 256 for i in ids)
    assert tok.decode(ids, skip_special_tokens=False) == text
    assert tok.pad_token == "<pad>"
    assert tok.eos_token == "<|endoftext|>"
    assert tok.vocab_size == 256 + 5


def test_hf_model_forward_shapes_and_state_sync(hf_config):
    from feather_v1.hf import FeatherV1ForCausalLM

    model = FeatherV1ForCausalLM(hf_config)
    ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    out = model(input_ids=ids)
    assert out.logits.shape == (1, 1, 64)
    assert torch.isfinite(out.logits).all()
    saved = np.array(model.feather._logit_projection)
    torch_proj = model.output.weight.detach().cpu().numpy().T  # (dim, vocab)
    assert np.allclose(saved, torch_proj, atol=1e-6)


def test_convert_core_to_hf_and_from_pretrained(tmp_path, hf_config):
    from feather_v1.hf import convert_core_to_hf
    from feather_v1.hf.modeling_feather_v1 import FeatherV1ForCausalLM as Cls

    core = make_core_config()
    from feather_v1 import FeatherV1Model

    m = FeatherV1Model(core)
    rng = np.random.default_rng(7)
    m.forward(rng.standard_normal((core.chunk_size, core.dim)))

    npz = tmp_path / "tiny.pt"
    m.save_weights(npz)
    bundle = convert_core_to_hf(npz, tmp_path / "bundle")
    assert (bundle / "config.json").is_file()
    state = bundle / "pytorch_model.bin"
    if not state.is_file():
        state = bundle / "model.safetensors"
        assert state.is_file(), "transformers state file missing"
    assert (bundle / "vocab.json").is_file()
    assert (bundle / "tokenizer_config.json").is_file()
    assert (bundle / "README.md").is_file()

    cfg_json = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    assert cfg_json["model_type"] == "feather"
    assert "FeatherV1ForCausalLM" in cfg_json["architectures"]
    assert "auto_map" in cfg_json

    loaded = Cls.from_pretrained(str(bundle))
    assert loaded.config.model_type == "feather"
    assert np.allclose(loaded.feather._logit_projection, m._logit_projection, atol=1e-4)


def test_auto_api_registration_when_available():
    pytest.importorskip("transformers")
    from feather_v1 import hf as hf_pkg

    assert hf_pkg.MODEL_TYPE == "feather"
    assert hf_pkg.AutoConfig is not None
    assert hf_pkg.AutoModelForCausalLM is not None
    assert hf_pkg.AutoTokenizer is not None
