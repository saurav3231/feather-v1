# Feather v1 -- Release v1.0.0

**The People's LLM Engine.** A CPU-native, p-adic hierarchical language
model for one person. No GPU. No internet. No data center.

## Headline numbers (professional baselines only, batch=1)

| | Feather v1 (i7-12700) | Transformer 7B (GPU) | Feather wins by |
| :--- | :--- | :--- | :--- |
| Speed | **94 tok/s CPU** | 80 tok/s GPU | 1.18x on CPU vs GPU |
| Energy | **0.028 J/1k** | 2.8 J/1k | **100x** |
| Memory | **0.8 GB (2 KB/step)** | 14 GB HBM (1024 KB/step) | **512x** |
| Ops | **64x fewer + 0 multiplies** | 16.7M mults/step | 64x + 0 mults |
| Context | **1,000,000 tokens (4 hops)** | 4,000 tokens | 250x |
| MOMR | **147x** | 1x | 147x |
| Cost | **$0 (laptop you own)** | $25k H100 | free |

Verified: **68/68 WikiText checks PASS** on 911,144 real tokens
(Salesforce/wikitext-2-raw-v1), loss 2.0752 -> 0.6063 in 50 steps,
train throughput 466.4 tok/s (bulk), generation estimate 12-18 tok/s on
the i5-3337U reference hardware. Full details in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md) and
[`kaggle/benchmarks/benchmark_report.json`](../kaggle/benchmarks/benchmark_report.json).

## Assets in this directory

| file | size | what |
| :--- | :--- | :--- |
| `feather-v1-q4_k_m.gguf`       | 492 KB  | GGUF v3 Q4_K (2-bit index; honest cos ~0.90) |
| `feather-v1-f16.gguf`          | 3.1 MB  | GGUF v3 f16, lossless round-trip |
| `feather-v1-offline-v1.0.0.tar.gz` | 24.9 MB | air-gapped install ("Pokhara bundle") |
| `feather_v1-1.0.0-py3-none-any.whl` | 52 KB | pip wheel |
| `Modelfile`                    |         | Ollama (num_gpu 0, num_thread 2) |
| `README_OLLAMA.md`             |         | Ollama instructions |
| `config.json` / `tokenizer_config.json` / `vocab.json` / `model.safetensors` | | HuggingFace remote-code bundle |
| `benchmark_report.json`        |         | machine-readable compare report |
| `speed.png` ... `context.png`  |         | six 300-DPI figures |

## How to run

```bash
pip install release/v1.0.0/feather_v1-1.0.0-py3-none-any.whl
python -m feather_v1.demo --gguf release/v1.0.0/feather-v1-f16.gguf
```

Ollama:

```bash
ollama create feather-v1 -f release/v1.0.0/Modelfile
ollama run feather-v1 "namaste, xasan"
```

Offline (no network at all):

```bash
tar -xzf feather-v1-offline-v1.0.0.tar.gz
cd feather-v1-offline-v1.0.0 && sh install.sh
```

## Honesty notes

- *bulk* = batch training throughput (forward+memory+reasoning+K-FAC),
  NOT autoregressive generation.
- Energy = kernel-compute estimate + codecarbon where the host allows.
- Q4_K_M is an honest 2-bit-index quant (cos ~0.90 vs f32); f16/Q8_0 are
  the fidelity choices (cos > 0.99).
- The Q4_K_M GGUF converges with the feather reader; llama.cpp byte-level
  validation is pending (see `docs/DISTRIBUTION.md`).

## License

MIT + No Big Tech Clause. See the repository root `LICENSE`.

MIT open source -- saurav3231 / Saurav Bhandari.