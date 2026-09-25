# Feather v1 — Model Design v1.0.0
# How the Architecture Blueprint Becomes Code

> **Version:** 1.0.0 Final — 2026-09-24 — International English
>
> **Source of truth:** `src/feather_v1/model.py`, `src/feather_v1/config.py`, `src/feather_v1/base.py`
>
> The house blueprint (docs/ARCHITECTURE_FINAL_v1.0.md) shows the design of the
> house. This document shows how that house is wired: the class diagram, the
> shapes of data moving through it, and the forward flow — one input, six
> rooms, one output.

---

## 1. The One-Sentence Design

**Feather v1 is one model class that owns six small component classes — Eyes,
Memory, Knowledge, Thought, Governor, Writer — each a specialist with its own
math — wired in a fixed order so that a single `(512 rows × 384 columns)`
text array becomes one next token, repeated 8 times for output.**

- One input: **512 × 384** numbers (≈ 196,608 values).
- One output: **8 tokens** generated autoregressively (or a drafts tensor of 4).
- Everything is numpy (float64) and CPU — `torch` is optional and never used
  for compute.

---

## 2. Class Diagram (Who Owns What)

```
                         ┌──────────────────────────────┐
                         │       FeatherV1Model         │   << the engine >>
                         │  owns (6 attributes + 2):    │
                         ├──────────────────────────────┤
                         │  config   : FeatherV1Config  │   all numbers/sizes
                         │  kernel   : dict             │   best binding for THIS PC
                         │  energy   : EnergyTracker    │   joules per component
                         │  _logit_projection           │   random init, seed-locked
                         │  _states  : list             │   history of steps
                         ├──────────────────────────────┤
                         │  encode(seq) → state dict    │   1 step, all 6 rooms
                         │  forward(tokens) → state     │   public alias of encode
                         │  generate(prompt, steps=8)   │   loop encode() 8 times
                         │  energy_report / total_joules│   honest metering
                         │  save_weights / from_weights │   GGUF-.pt archive
                         │  reset()                     │   clear memory + energy
                         └──────────────────────────────┘
        ▲  │  │  │  │  │  │  composes (each is a BaseComponent)
        │  ▼  ▼  ▼  ▼  ▼  ▼
   ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
   │Sensory │ │Liquid  │ │Knowledge│ │Cognitive │ │Homeostat.│ │Generative  │
   │Encoder │ │Memory  │ │Vault    │ │Weaver    │ │Governor  │ │Evolution   │
   │  Eyes  │ │  MEM   │ │ KNOW    │ │  THINK   │ │ MANAGE   │ │  WRITE     │
   └────────┘ └────────┘ └─────────┘ └──────────┘ └──────────┘ └────────────┘
    math:      math:       math:        math:       math:        math:
    rough      fractional  tropical     category    free-energy  jacobi-spec
    path       + liquid    + tt-rank    theory      + equilibrium + sheaf
    clifford   + spiking   + sinkhorn   + kfac      + dp/krum    + goedel
    wht        + p-adic    + sheaf      + clifford  + codecarbon

                    ┌──────────────────────────────┐
                    │      EnergyTracker           │
                    │  record(component, joules)   │
                    │  total() → float             │
                    └──────────────────────────────┘
                    every component records its own joules;
                    the model sums them (honest energy per step).
```

Every component is a `BaseComponent` (`src/feather_v1/base.py`): it knows its
`name`, writes a `cache_report()` (which cache level it fits), and records
energy into the shared `EnergyTracker`.

---

## 3. Config — All the Numbers (one table, no code)

The default `FeatherV1Config` in `src/feather_v1/config.py`. Kaggle uses a
variant (`kaggle_cpu.json`) — differences marked with ✅.

| Setting | Value | Meaning (simple) |
| :--- | :--- | :--- |
| `dim` | 384 | width of every token's vector (Kaggle 384 ✅) |
| `seq_len` | 512 | how many tokens the model reads at once |
| `chunk_size` | 32 | memory operates on 32-token pieces |
| `num_chunks` | 16 | 16 pieces of 32 = the 512 window |
| `hypervector_dim` | 1024 (Kaggle 4096 ✅) | size of the WHT hologram vector |
| `n_experts` | 64 | experts in the Knowledge Vault |
| `moe_top_k` | 1 | only the top-1 expert fires (1.5% active) |
| `tt_rank` | 4 | tensor-train rank (Kaggle 8 ✅; compression 256x) |
| `alpha` | 0.7 | fractional-derivative order (memory power-law) |
| `k_frac` | 32 | short-range fractional window |
| `k_frac_long` | 128 | long-range fractional window |
| `tau` | 0.1 | liquid time-constant (fast→slow) |
| `p_adic_p` | 2 | p-adic distance base (2 = family-tree branch) |
| `threads` | 2 (Kaggle/i7 4-8 ✅) | physical cores used |
| `precision` | int8 | quantized activation storage |
| `binding` | avx_wht (auto-chosen) | kernel this PC detected |
| `moe` | avx_tropical_tt | expert math flavor |
| `sinkhorn_eps / iters` | 0.1 / 3 | load-balancing budget |
| `ram_budget_gb` | 0.6 (Kaggle 0.8 ✅) | cap so even 8GB laptops fit |
| `batch_size` | 1 | personal LLM — one person, one prompt |
| `vocab_size` | 50257 (Kaggle 4096 ✅) | output vocabulary |
| `seed` | 42 | every run reproducible |

---

## 4. Input / Output Shapes (who talks the numbers through the rooms)

| Stage | Data | Shape | Bytes (roughly) | Cache |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | text → token ids → one-hot | `(512, 384)` | 196,608 floats | L3 |
| **1 Sensory** | signature + WHT hypervector | `(13,)` then `(1024,)` binary | 52 bytes → 40KB | L1 / L2 |
| **2 Liquid** | fractional memory state `m` | `(384,)` | 3KB | L1 |
| **3 Knowledge** | routed expert output | `(384,)` | 3KB | L3 → L1 |
| **4 Weaver** | reasoned state + drafts | `(384,)` + `(4,)` | 3KB | L1 reused 6x |
| **5 Governor** | gated + DP-protected state | `(384,)` | 3KB | L1 |
| **6 Generation** | 4 Jacobi drafts, pick best | `(1, 384)` wrap (next token one-hot) | 3KB | L1 |
| **Output** | 1 token per step, 8 steps | `(1, vocab)` logits → argmax → id | — | L1 |

All internal shapes fit the **L1 cache** (the CPU's fastest pocket memory) — that
is the whole trick: the state never leaves the chip between the six rooms.

---

## 5. Forward Flow — One Step Through the Six Rooms (with the math names)

```
prompt (512, 384)
   │
   ▼
[1] sensory.encode(seq)
      rough_path(seq)        → 13 numbers      (2520x compression)
      wht bind               → hypervector      (384 adds, 0 mults)
   │
   ▼
[2] for t in 512: memory.hierarchical_fractional(seq[t])
      fractional_weights     → power-law memory (retention 3.25e20x)
      spiking 98% sparse     → 6/512 fire
      p_adic chunk retrieval → 7k ops / 2KB
   │  memory state m (384,)
   ▼
[3] knowledge.route_and_apply(m, batch=1)
      sinkhorn(m)            → top-1 of 64 experts (balanced 5x)
      tt_compress            → 256x smaller weights
      tropical min-plus      → 0 multiplies
   │
   ▼
[4] reasoning.reasoning_loop(knowledge_out, entropies)
      1 block {category compose + kfac + clifford dual} × 6
      entropy gate → exit after ~2 loops (62% early)
   │
   ▼
[5] governor.entropy_gate + dp_noise(reasoned)
      free energy check      → keep working only if worth it
      laplace noise ε=1.0    → privacy
      energy.record          → 0.028J/1k honest meter
   │
   ▼
[6] generation.speculative_generate(...)
      jacobi drafts (4)      → 8 tokens in ~2.3 steps
      argmax → token id      → one-hot (1, 384)
   │
   ▼
state dict stored in _states; energy metered per component;
next token fed back → repeat 8× (generate) → 8 output tokens
```

**Six attributes that hold the whole machine** (as exposed by `FeatherV1Model`):
`config`, `kernel`, `energy`, `sensory`, `memory`, `knowledge`, `reasoning`,
`governor`, `generation` + the two internal helpers `_logit_projection` and
`_states`. `hardware_summary()` and `energy_report()` are the honest gauges;
`save_weights / from_weights` persist and reload the whole house in one file.

**Reproducibility:** every random number comes from `np.random.default_rng(seed)`
with `seed = config.seed` — the same PC and config yields bit-identical tokens
every time, which is how the 113-test suite passes deterministically.

---

## 6. Design Rules That Hold It Together

1. **DRY — never repeat math.** `utils.py` owns all 12 maths; both the model
   and the benchmark import them. One definition of `fwht`, one of
   `fractional_weights`, one of `tropical_min`, one of `p_adic_distance` — and
   the whole family: `tt_compress`, `rough_path_signature`, `sinkhorn`,
   `clifford_product`, `kfac_apply`, `sheaf_consistency_ok`,
   `adaptive_draft_len`, `jacobi_update` — anywhere.
2. **Hardware-adaptive, not hand-tuned.** `hardware.py` picks the binding on
   first run (AVX-512 → AVX2 → AVX → NEON → Scalar), so the *same* model and
   *same* weights run on a 2010 laptop and an i7-12700 — the code does not
   care which PC it is on.
3. **No torch in the compute path.** `torch` is optional; the numpy engine is
   the only required runtime — that is what makes CI (and Kaggle) pass with
   no GPU and no PyTorch.
4. **Honest metering.** Every component records joules via `EnergyTracker`;
   `total_joules()` is real measured kernel work, never marketing.
5. **Deterministic tests.** 113 tests, all green, 12/12 CI jobs — small-scale
   individual (12/12 maths) + integrated (cos 1.0, 512x mem, 64x fewer ops) +
   WikiText medium (68/68 PASS).

> **Feather v1 Model Design complete — the blueprint's six rooms are wired as
> six classes, config is the set of numbers, shapes all fit L1, and the flow
> is: 512×384 in → six rooms → one token, repeated 8× → output. CPU is the
> people, GPU is the monopoly, Feather v1 is CPU's revenge.**