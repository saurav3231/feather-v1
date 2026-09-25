"""Feather v1 -- distribution phase end-to-end tests.

Cover the offline bundle (air-gap install shape), the Docker/compose
recipe, the ``gguf_convert`` CLI, and -- most importantly -- that the core
engine still works with the network stack completely disabled (the whole
point of the Pokhara / airplane-mode distribution story).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

import feather_v1

REPO_ROOT = Path(feather_v1.__file__).resolve().parent.parent.parent


def _tiny_weights(path: Path) -> None:
    import json

    import numpy as np

    from feather_v1.config import FeatherV1Config as CoreConfig

    core = CoreConfig(dim=32, seq_len=64, chunk_size=8, num_chunks=8, vocab_size=32)
    with open(path, "wb") as fh:
        np.savez(
            fh,
            config=json.dumps(core.to_dict()).encode("utf-8"),
            logit_projection=(
                np.random.default_rng(23).standard_normal((32, 32)).astype(np.float32)
                / 8
            ),
        )


def test_offline_bundle_structure_builds(tmp_path):
    weights = tmp_path / "tiny.pt"
    _tiny_weights(weights)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "offline_installer.py"),
            "--no-wheels",
            "--quant",
            "q4_k_m",
            "--weights",
            str(weights),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    tarball = REPO_ROOT / "dist" / "feather-v1-offline-v1.0.0.tar.gz"
    assert tarball.is_file()
    with tarfile.open(tarball) as tf:
        names = tf.getnames()
    joined = "\n".join(names)
    assert any("install.sh" in n for n in names)
    assert any("feather-v1-q4_k_m.gguf" in n for n in names)
    assert any("feather-v1-kaggle.pt" in n for n in names)
    assert "configs/" in joined and "README.txt" in joined


def test_dockerfile_slim_cpu_only():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.10-slim" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "COPY scripts/serve.py ./serve.py" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "cuda" not in dockerfile.lower()


def test_docker_compose_exposes_api():
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "feather:" in compose
    assert '"8000:8000"' in compose
    assert "mem_limit: 2g" in compose


def test_gguf_convert_cli_writes_valid_file(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    npz = tmp_path / "w.pt"
    import json

    import numpy as np

    from feather_v1.config import FeatherV1Config as CoreConfig

    core = CoreConfig(dim=32, seq_len=64, chunk_size=8, num_chunks=8, vocab_size=32)
    with open(npz, "wb") as fh:
        np.savez(
            fh,
            config=json.dumps(core.to_dict()).encode("utf-8"),
            logit_projection=np.zeros((32, 32), dtype=np.float32),
        )
    out = tmp_path / "out.gguf"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "gguf_convert.py"),
            "--input",
            str(npz),
            "--output",
            str(out),
            "--quant",
            "q8_0",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    from feather_v1.gguf import read_gguf

    assert read_gguf(out).version == 3


def test_core_works_with_network_disabled(tmp_path):
    """Airplane mode: forward inference never needs a socket."""
    code = (
        "import numpy as np\n"
        "import socket\n"
        "class NoNet(socket.socket):\n"
        "    def connect(self, *a): raise RuntimeError('offline')\n"
        "    def send(self, *a): raise RuntimeError('offline')\n"
        "socket.socket = NoNet\n"
        "from feather_v1 import FeatherV1Config, FeatherV1Model\n"
        "cfg = FeatherV1Config(dim=16, seq_len=16, chunk_size=8, num_chunks=8, vocab_size=32)\n"
        "m = FeatherV1Model(cfg)\n"
        "out = m.forward(np.zeros((4, 16)))\n"
        "assert out['final_output'].shape == (1, 16)\n"
        "print('offline-forward-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "offline-forward-ok" in result.stdout
