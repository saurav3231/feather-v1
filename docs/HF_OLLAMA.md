# HuggingFace + Ollama integration notes

## What the HF package is (and is not)

`src/feather_v1/hf/` is a *container*: the numpy engine
(`feather_v1.model.FeatherV1Model`) stays the single source of truth; torch
appears only as the reference-counting layer transformers needs.

- `configuration_feather_v1.py` -> `FeatherV1Config(PretrainedConfig)`;
  `model_type="feather"`; field names mirror the numpy config's `to_dict()`
  camelCase keys exactly, and `to_core()` rebuilds the numpy config 1:1.
- `tokenization_feather_v1.py` -> `FeatherV1Tokenizer(PreTrainedTokenizer)`:
  byte-level (each token is one UTF-8 byte), reversible, Nepali-safe, no
  external tokenizer lib. Reuses `feather_v1.utils.byte_tokenize`.
- `modeling_feather_v1.py` -> `FeatherV1ForCausalLM(PreTrainedModel)`: wraps
  the numpy model; `output.weight` mirrors `model._logit_projection` and is
  re-synced on every `load_state_dict`.

Importing `feather_v1.hf` registers the three classes with the transformers
Auto API (`AutoConfig` / `AutoModelForCausalLM` / `AutoTokenizer`) for
`model_type="feather"`.

```python
from feather_v1.hf import FeatherV1Config, FeatherV1ForCausalLM

cfg = FeatherV1Config.from_core(core_config)          # numpy config in
model = FeatherV1ForCausalLM(cfg)                     # torch shell
out = model(input_ids=torch.tensor([[5, 6, 7]]))
```

## Converting numpy weights -> HF bundle

```bash
python scripts/hf_upload.py --build-only
# writes dist/hf/{config.json, pytorch_model.bin, vocab.json,
#                tokenizer_config.json, README.md}
```

config.json contains `model_type: feather`, `architectures:
[FeatherV1ForCausalLM]` and `auto_map` so `trust_remote_code=True` loads work
even without the package pre-installed.

## Ollama

- `ollama/Modelfile`: `FROM` a feather GGUF, `num_thread 2`, `num_gpu 0`
  (CPU-native), tensor chat template whose special tokens
  (`<|system|>/<|user|>/<|assistant|>`) are declared in the tokenizer.
- `scripts/ollama_publish.py`: converts `models/kaggle/feather-v1-kaggle.pt`
  to the chosen quant GGUF then runs `ollama create feather-v1 -f
  ollama/Modelfile`. Without the binary it prints the exact manual commands.

```bash
python scripts/ollama_publish.py --quant q8_0
ollama run feather-v1 "namaste, xasan"
```

## DRY rules that produced this layout

All math lives once in `src/feather_v1/utils.py` (fwht, fractional_weights,
sinkhorn, tropical_min, p_adic_distance, tt_compress, rough_path_signature,
clifford_product, kfac_fisher_approx, kfac_apply, sheaf_consistency_check,
byte_tokenize). The HF and GGUF layers import, never copy. The GGUF
writer/reader pair in `src/feather_v1/gguf/` is the single implementation of
the 24-byte-header v3 layout shared with the Kaggle notebook writer.