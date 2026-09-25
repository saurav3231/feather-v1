# Feather v1 -- Paper & Release (v1.0.0)

This directory is the arXiv-ready paper plus the built release assets for
**v1.0.0**.

## Paper

| file | what |
| :--- | :--- |
| `paper/main.tex`     | arXiv LaTeX source (~10 pages, `bibtex`-ready) |
| `paper/main.pdf`     | rendered PDF (reportlab build, run `python paper/build_pdf.py`) |
| `paper/references.bib` | 18 references |
| `paper/figures/`     | six 300-DPI PNGs (speed / energy / memory / ops / momr / context) |
| `paper/build_pdf.py` | regenerates `main.pdf` without pdflatex |
| `paper/README_PAPER.md` | this file |

`main.pdf` is built with reportlab because this authoring host has no
pdflatex; `main.tex` is a normal arXiv submission that compiles with
`pdflatex paper/main.tex` on any machine that has TeX. Content is
section-for-section identical.

## Release v1.0.0 (`release/v1.0.0/`)

| file | size | what |
| :--- | :--- | :--- |
| `feather-v1-q4_k_m.gguf`         | 492 KB  | GGUF v3, 2-bit-index Q4_K quantization (honest cos ~0.90) |
| `feather-v1-f16.gguf`            | 3.1 MB  | GGUF v3 float16, lossless round-trip |
| `feather-v1-offline-v1.0.0.tar.gz` | 24.9 MB | "Pokhara bundle", fully offline install |
| `feather_v1-1.0.0-py3-none-any.whl` | 52 KB | pip wheel |
| `Modelfile` / `README_OLLAMA.md` |         | Ollama num_gpu 0 |
| `config.json`, `tokenizer_config.json`, `vocab.json`, `model.safetensors` | | HuggingFace bundle |
| `benchmark_report.json`          |         | machine-readable compare report |
| 6 PNGs                           |         | 300-DPI figures |

Reproduce everything:

```bash
python kaggle/scripts/kaggle_benchmark.py          # report + charts
python scripts/gguf_convert.py --quant f16         # GGUF f16
python scripts/gguf_convert.py --quant q4_k_m      # GGUF Q4_K_M
python scripts/offline_installer.py --no-wheels    # offline tarball
python paper/build_pdf.py                          # paper PDF
```

## Live publishing (needs credentials this repo does not have)

```bash
pip install build twine                          # PyPI
python -m twine upload release/v1.0.0/*.whl      # needs PYPI_TOKEN
HF_TOKEN=hf_... python scripts/hf_upload.py      # HuggingFace
ollama create feather-v1 -f release/v1.0.0/Modelfile   # needs ollama CLI
```

The build scripts run and print the exact manual command when credentials
are missing.