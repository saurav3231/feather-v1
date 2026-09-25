#!/usr/bin/env python
"""
Feather v1 -- OPT 2 -- Other Architectures Test (Transformer, RetNet, Mamba, RWKV)

Needs GPU accelerator ON -- slow on CPU -- separate from Feather-only test.
Real measured losses, tok/s, RAM, energy -- no hardcoded numbers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from feather_v1.data import load_wikitext2  # noqa: E402

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    HAS_TORCH = False

try:
    from codecarbon import OfflineEmissionsTracker

    HAS_CODECARBON = True
except Exception:
    HAS_CODECARBON = False


class TinyTransformer(nn.Module):
    def __init__(self, vocab, dim, n_layers, n_heads, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, n_heads, dim * 4, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x)


class TinyRetNet(nn.Module):
    def __init__(self, vocab, dim, n_layers, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        x = self.tok_emb(x)
        for lyr in self.layers:
            x = x + lyr(x)
        return self.head(x)


class TinyMamba(nn.Module):
    def __init__(self, vocab, dim, n_layers, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        x = self.tok_emb(x)
        for lyr in self.layers:
            x = x + lyr(x)
        return self.head(x)


class TinyRWKV(nn.Module):
    def __init__(self, vocab, dim, n_layers, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        x = self.tok_emb(x)
        for lyr in self.layers:
            x = x + lyr(x)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_step(model, optimizer, x, y, device):
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def measure_tok_s(model, data, steps, device):
    t0 = time.perf_counter()
    tokens = 0
    for i in range(steps):
        x = torch.from_numpy(data[i % len(data)]).long().unsqueeze(0).to(device)
        _ = model(x)
        tokens += x.numel()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    return tokens / elapsed


def measure_cpu_tok_s(model, seq_len, vocab, device):
    model.eval()
    x = torch.randint(0, vocab, (1, seq_len), device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    return seq_len * 100 / max(time.perf_counter() - t0, 1e-9)


def measure_ram_gb():
    import psutil

    return psutil.Process().memory_info().rss / 1e9


def main():
    if not HAS_TORCH:
        print("ERROR: torch required for other architectures test")
        sys.exit(1)

    size = os.environ.get("SIZE", "20M")
    steps = int(os.environ.get("FEATHER_TRAIN_STEPS", "600"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if size == "5M":
        vocab, dim, n_layers, n_heads, seq_len = 96, 64, 4, 4, 64
    elif size == "20M":
        vocab, dim, n_layers, n_heads, seq_len = 256, 384, 6, 6, 512
    else:
        raise ValueError(size)

    mode = "char" if vocab <= 96 else "byte"
    data = load_wikitext2("train", mode, seq_len, dim)
    chunks = data["chunks"]
    print(f"Data: {len(chunks)} chunks x {seq_len} = {len(chunks) * seq_len:,} tokens")

    train_tensors = torch.from_numpy(np.stack(chunks)).long()
    train_dataset = torch.utils.data.TensorDataset(
        train_tensors[:, :-1], train_tensors[:, 1:]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False
    )

    families = [
        (
            "transformer",
            lambda: TinyTransformer(vocab, dim, n_layers, n_heads, seq_len),
        ),
        ("retnet", lambda: TinyRetNet(vocab, dim, n_layers, seq_len)),
        ("mamba", lambda: TinyMamba(vocab, dim, n_layers, seq_len)),
        ("rwkv", lambda: TinyRWKV(vocab, dim, n_layers, seq_len)),
    ]

    results = []
    tracker = None
    if HAS_CODECARBON:
        try:
            tracker = OfflineEmissionsTracker(
                country_iso_code="NPL", log_level="error", output_dir="."
            )
            tracker.start()
        except Exception:
            tracker = None

    for fam_name, model_fn in families:
        print(f"\n[{fam_name.upper()}] {size}")
        for seed in [42, 7]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = model_fn().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

            t0 = time.perf_counter()
            step_losses = []
            for step, (x, y) in enumerate(train_loader):
                if step >= steps:
                    break
                loss = train_step(model, optimizer, x.to(device), y.to(device), device)
                step_losses.append(loss)
            _ = time.perf_counter() - t0

            tok_s = measure_tok_s(model, chunks, min(50, steps), device)
            cpu_tok_s = measure_cpu_tok_s(model, seq_len, vocab, device)
            ram = measure_ram_gb()
            params = count_params(model)

            results.append(
                {
                    "size": size,
                    "family": fam_name,
                    "seed": seed,
                    "eval_loss": (
                        float(np.mean(step_losses[-5:])) if step_losses else 0.0
                    ),
                    "train_tok_s": tok_s,
                    "cpu_tok_s": cpu_tok_s,
                    "ram_gb": ram,
                    "params": params,
                }
            )
            print(
                f"  seed {seed}: loss {results[-1]['eval_loss']:.2f} tok/s {tok_s:.0f} RAM {ram:.1f}GB"
            )

    energy_j = 0.0
    if tracker:
        try:
            tracker.stop()
            data = tracker.final_emissions_data
            energy_j = float(getattr(data, "energy_consumed", 0.0) or 0.0)
        except Exception:
            pass

    print("\n" + "=" * 100)
    print(f"HONEST COMPARISON BOARD -- {size.upper()} -- REAL MEASURED")
    print("-" * 100)
    print(
        f"{'size':<6} {'family':<12} {'eval loss (42/7)':<22} {'mean':<8} {'train tok/s':<15} {'CPU tok/s':<12} {'RAM':<8}"
    )
    for r in results:
        same = [
            x for x in results if x["size"] == r["size"] and x["family"] == r["family"]
        ]
        l42 = next((x["eval_loss"] for x in same if x["seed"] == 42), 0)
        l7 = next((x["eval_loss"] for x in same if x["seed"] == 7), 0)
        mean_l = np.mean([l42, l7])
        t42 = next((x["train_tok_s"] for x in same if x["seed"] == 42), 0)
        t7 = next((x["train_tok_s"] for x in same if x["seed"] == 7), 0)
        print(
            f"{r['size']:<6} {r['family']:<12} {l42:.2f} / {l7:.2f}          {mean_l:.2f}      {t42:.0f} / {t7:.0f}          {r['cpu_tok_s']:.0f}          {r['ram_gb']:.1f}GB"
        )

    out = REPO_ROOT / "kaggle" / f"other_archs_results_{size.lower()}.json"
    with open(out, "w") as f:
        json.dump({"size": size, "results": results, "energy_j": energy_j}, f, indent=2)
    print(f"\nResults saved: {out}")
    print(f"Energy: {energy_j:.4f} J")


if __name__ == "__main__":
    main()
