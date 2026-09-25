# Feather v1 on Ollama

Feather v1 ships as a quantized GGUF (v3) that Ollama can load the same way
it loads llama/llava files. The engine is open-source Python (numpy-only),
but the GGUF "runtime" here is the standard llama.cpp-based loading path.

## Build the model in Ollama

```bash
# 1. convert numpy weights -> Q4_K_M GGUF (already committed at
#    models/kaggle/feather-v1-kaggle.gguf for f16; regenerate any quant)
python scripts/gguf_convert.py \
  --input models/kaggle/feather-v1-kaggle.pt \
  --output dist/feather-v1-q4_k_m.gguf \
  --quant q4_k_m

# 2. create + run (requires the `ollama` binary locally)
ollama create feather-v1 -f ollama/Modelfile
ollama run feather-v1 "namaste, xasan"
```

## Status of GGUF quants

| quant  | llama.cpp byte-compat | quality (cos of logit projection) | size   |
|--------|-----------------------|-----------------------------------|--------|
| F32    | verified (writer)     | 1.0                               | 6.3 MB |
| F16    | verified (writer)     | 1.0                               | 3.2 MB |
| Q8_0   | verified (writer)     | >0.99                             | 1.6 MB |
| Q4_K_M | pending llama.cpp     | ~0.8 (2-bit index budget)         | 0.9 MB |

`pip install feather-v1` users get the numpy engine; the GGUF file is the
portable format for Ollama. `python scripts/ollama_publish.py` builds the
converted GGUF then runs `ollama create` (requires the ollama CLI).

## Offline / airplane mode

A GGUF file + the `feather-v1` wheel + `--no-index` install is fully offline;
see `scripts/offline_installer.py`.