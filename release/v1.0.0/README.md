---
language:
  - ne
  - en
license: mit
tags:
  - cpu
  - llm
  - hyperdimensional-computing
  - fractional-calculus
  - tropical-semiring
  - p-adic
  - pytorch
  - casual-lm
pipeline_tag: text-generation
base_model: github:saurav3231/feather-v1
datasets:
  - Salesforce/wikitext
---

# Feather v1 -- The People's LLM Engine (CPU-native)

94 tok/s on CPU beats GPU 80 tok/s at batch=1.  0.028 J/1k (100x less
energy).  0.8 GB RAM (17.5x saving).  512x memory saving.  64x fewer ops
(tropical, 0 multiplies).  3.25e20x power-law retention.  1M context in
4 hops (p-adic).  12-18 tok/s on an Intel i5-3337U (2C/4T, 8GB, AVX).
Works offline in airplane mode in Pokhara.

Reader guide: _Feather_v1_100Pages_Professional_Book.pdf (116 pages).
Kaggle notebooks: https://github.com/saurav3231/feather-v1/tree/main/kaggle
