# Distribution Guide (HF / Ollama / GGUF / pip / Docker / offline)

Feather v1 publishes through five channels; all five use the same numpy
weights (`models/kaggle/feather-v1-kaggle.pt`).

## 1. pip (PyPI wheel)

```bash
python -m build            # builds sdist + wheel under dist/
python -m twine upload dist/*   # needs PYPI_TOKEN
```

The wheel always carries the CPU numpy engine with zero hard deps beyond
numpy + py-cpuinfo. Torch / transformers / flask are extras:

```bash
pip install feather-v1             # everything, offline-safe
pip install 'feather-v1[hf]'       # + HuggingFace container
pip install 'feather-v1[serve]'    # + Flask HTTP API
```

## 2. HuggingFace Hub

`scripts/hf_upload.py` converts the npz into a `from_pretrained`-loadable
bundle (config.json with `model_type="feather"` + auto_map, pytorch_model.bin,
vocab.json, tokenizer_config.json, model card) in `dist/hf/` and publishes
to `saurav3231/feather-v1`.

```bash
python scripts/hf_upload.py --build-only   # build the bundle locally
HF_TOKEN=... python scripts/hf_upload.py   # build + publish
```

Loading on a machine without `feather_v1` installed:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("saurav3231/feather-v1", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("saurav3231/feather-v1", trust_remote_code=True)
```

`trust_remote_code=True` is required when the feather package is not already
installed (auto_map points the loader at the bundled classes); with the pip
package installed the classes are already registered and no remote code runs.

## 3. Ollama / llama.cpp (GGUF v3)

`scripts/gguf_convert.py` quantizes the 384x4096 logit projection:

| quant  | tensor type | llama.cpp byte-compat | size   |
|--------|-------------|-----------------------|--------|
| F32    | 0           | verified (writer)     | 6.3 MB |
| F16    | 1           | verified (writer)     | 3.2 MB |
| Q8_0   | 8           | verified (writer)     | 1.6 MB |
| Q4_K_M | 12 (Q4_K)   | **pending**           | 0.9 MB |

Important honesty note: the Q4_K_M 2-bit-indexed quantization of a *random
normal* matrix reaches cos ~0.8, not 0.95 - the 2-bit index budget physically
caps 4-level-per-block fidelity. The checklist's blanket "cos > 0.95 after
quant" is met by F16 and Q8_0 (cos > 0.99). The feather reader round-trips
Q4_K_M exactly (writer/reader inverse pairs), but byte-identity with llama.cpp
should be validated on a machine that has the `llama.cpp` binary before the
Q4_K_M GGUF is treated as the canonical Ollama artifact; Q8_0 is the current
recommended default for Ollama.

```bash
python scripts/gguf_convert.py --quant q8_0 --output dist/feather-v1-q8_0.gguf
ollama create feather-v1 -f ollama/Modelfile   # FROM the q8_0 gguf
ollama run feather-v1 "namaste, xasan"
```

## 4. Docker (CPU-only server)

```bash
docker compose up --build
curl -s localhost:8000/completion -H 'content-type: application/json' \
     -d '{"prompt": "namaste, xasan", "steps": 16}'
# -> {"prompt": "...", "steps": 16, "text": "..."}
curl -s localhost:8000/healthz
```

Image: `python:3.10-slim` (no CUDA by design); the model weights are baked
in at build time; the container opens no outbound sockets (airplane mode).

## 5. Offline installer ("Pokhara bundle")

`scripts/offline_installer.py` builds `dist/feather-v1-offline-v1.0.0.tar.gz`:

```
feather-v1-offline-v1.0.0/
├── install.sh                # pip install --no-index --find-links=wheels
├── wheels/                   # feather_v1 wheel (+ numpy, py-cpuinfo)
├── feather-v1-q4_k_m.gguf    # Ollama-format weights
├── feather-v1-kaggle.pt      # numpy engine weights
├── configs/                  # i5_3337U + kaggle_cpu tuning
└── README.txt
```

Transfer the tarball to the air-gapped machine, run `sh install.sh`, and the
numpy engine runs entirely offline (verified by the socket-blocked test in
`tests/test_distribution.py`).

## Publishing checklist & live status

| channel           | artifact                                | status        |
|-------------------|-----------------------------------------|---------------|
| GitHub            | `main` branch + CI (12 jobs)            | done (push)   |
| PyPI              | `feather_v1-1.0.0` wheel                | needs PYPI_TOKEN |
| HuggingFace       | `saurav3231/feather-v1` model repo      | needs HF_TOKEN |
| Ollama            | `saurav3231/feather-v1`                  | needs ollama CLI |
| Docker Hub        | build + push from Dockerfile            | needs docker login + buildx |

Live publishes to PyPI / HF / Ollama / Docker Hub require credentials the CI
does not have; the build scripts above run and print the exact manual command
when credentials are missing.