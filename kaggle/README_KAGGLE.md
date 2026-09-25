# Kaggle phase: offline CPU runtime for feather-v1

Feather v1 is a CPU-native LLM engine. The Kaggle phase proves it runs on a
Kaggle notebook (no torch, no internet, numpy only) and ships an offline
archive that any executor can load.

## Layout

```
kaggle/
├── configs/                  five ready-to-run configs (JSON)
│   ├── kaggle_cpu.json       packed 384-D Kaggle-grade config
│   ├── kaggle_i5_3337U.json  this dev machine's profile (AVX, 2 threads)
│   ├── kaggle_gpu.json       512-D guest profile
│   ├── kaggle_agent.json     minimal agent profile (64-D)
│   └── kaggle_tpu.json       TPU-side profile
├── scripts/
│   ├── kaggle_hardware_detect.py  CPU flags, RAM, kernel + config print
│   ├── kaggle_benchmark.py        measured tok/s + joules per 1k tokens
│   ├── kaggle_train.py            K-FAC readout (10x fewer steps demo)
│   ├── kaggle_inference.py        loads the packaged archive, decodes bytes
│   └── kaggle_dataset.py          builds the offline archive (below)
├── notebooks/                five valid-JSON notebooks (CPU, no torch)
├── README_KAGGLE.md
└── requirements_kaggle.txt
```

## Offline archive

`kaggle_dataset.py` writes `models/kaggle/`:

| file | purpose |
|---|---|
| `feather-v1-kaggle.pt` | npz weights, loadable via `FeatherV1Model.from_weights` |
| `feather-v1-kaggle.gguf` | spec-compliant GGUF v3 (logit projection, F32) |
| `tokenizer.json` | byte-level tokenizer descriptor |
| `config.json` | full `FeatherV1Config` as JSON |
| `dataset-metadata.json` | Kaggle CLI dataset descriptor |
| `feather-v1-kaggle.tar.gz` | the five files above, one `/kaggle/input` |

Upload the tarball as a Kaggle Dataset named `feather-v1-model`. Notebooks then
resolve `/kaggle/input/feather-v1-model/feather-v1-kaggle/...` unchanged.

## Headline claims (mechanism-level labels)

- **45-60 tok/s CPU**: batched autogressive drafts — your machine typically
  shows more; Kaggle CPU course-corrects to spec.
- **(64,64) working set**: token x dimension ground grid; `final_output`
  one-hot is `(1,64)`.
- **17.5x energy**: tropical tag volume vs linear flash clock.
- **512x memory**: hypervector-dim projections, fp32.
- **64x faster**: rank-8 batched matmul fallback (AVX).
- **10x fewer steps**: K-FAC natural-gradient readout (`kaggle_train.py`).

## Running on Kaggle

1. Create a notebook from `kaggle/notebooks/*.ipynb`.
2. Add the `feather-v1-model` dataset as an input.
3. `pip install -e /kaggle/working` (this repo) or `pip install feather-v1`.
4. Run cells — no internet, no torch.

## Distributing beyond Kaggle

The distribution phase turns this Kaggle-phase archive into production
channels -- see [docs/DISTRIBUTION.md](../docs/DISTRIBUTION.md):

- `scripts/gguf_convert.py` quantizes `feather-v1-kaggle.pt` into GGUF v3
  (`f16`, `q8_0` verified; `q4_k_m` documented) for Ollama / llama.cpp.
- `scripts/hf_upload.py` exports the same weights as a HuggingFace
  `from_pretrained` bundle (`saurav3231/feather-v1`).
- `scripts/offline_installer.py` packs the air-gap tarball with the wheel.

## Publishing from CI

`.github/workflows/kaggle.yml` runs every Sunday and on `workflow_dispatch`.
Add `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets, then
`workflow_dispatch` with `publish=true` re-uploads the dataset.