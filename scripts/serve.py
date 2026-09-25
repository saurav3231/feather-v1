#!/usr/bin/env python
"""Tiny CPU-only completion server used by the Docker image.

    docker compose up --build
    curl -s localhost:8000/completion -H 'content-type: application/json' \
         -d '{"prompt": "namaste, xasan"}'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from feather_v1 import FeatherV1Model  # noqa: E402
from feather_v1.utils import byte_tokenize  # noqa: E402

try:
    from flask import Flask, jsonify, request  # type: ignore
except Exception:  # pragma: no cover - optional server dep
    print(
        "flask not installed; run `pip install flask` or use the numpy CLI: "
        "python scripts/offline_installer.py --serve"
    )
    raise SystemExit(2) from None

_MODEL = os.environ.get("FEATHER_MODEL", str(REPO_ROOT / "model.pt"))
app = Flask(__name__)

model = None


def get_model() -> FeatherV1Model:
    global model  # noqa: PLW0603
    if model is None:
        model = FeatherV1Model.from_weights(_MODEL)
    return model


@app.post("/completion")
def completion() -> Flask.response_class:
    payload = request.get_json(force=True) or {}
    prompt = str(payload.get("prompt", ""))
    steps = int(payload.get("steps", 16))
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    tokens = byte_tokenize(prompt)
    seq = get_model().generate(tokens, steps=steps)
    text = bytes(int(b) for b in seq).decode("utf-8", errors="replace")
    return jsonify({"prompt": prompt, "steps": steps, "text": text})


@app.get("/healthz")
def healthz() -> Flask.response_class:
    return jsonify({"ok": True, "model": str(_MODEL)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
