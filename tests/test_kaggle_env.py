"""Feather v1 — Kaggle environment adaptation tests.

The adaptation layer must be correct, offline-safe, and never import torch
(CPU/agent-safe runtime). All tests pass on any host: they only assert the
detection contract, not a specific machine. Inside a real Kaggle runtime the
same tests still pass because the detection is data-driven.
"""

import json
import os
from pathlib import Path

from feather_v1 import (
    FeatherV1Config,
    FeatherV1Model,
    detect_cpu_features,
    get_best_kernel,
    is_kaggle,
    kaggle_env,
)
from feather_v1.hardware import cuda_device_summary, has_internet
from tests.conftest import make_config

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_CONFIGS = sorted((REPO_ROOT / "kaggle" / "configs").glob("*.json"))

REQUIRED_SKELETON_KEYS = [
    "model_name",
    "version",
    "description",
    "hardware_target",
    "harness",
    "ladder",
    "pipeline",
    "neurosymbolic",
    "hidden_markov",
    "prediction_head",
    "stdout_view",
    "matplotlib_surface",
]


def _has_kaggle_dirs() -> bool:
    return os.path.isdir("/kaggle/working") or os.path.isdir("/kaggle/input")


def test_is_kaggle_tracks_env_var(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    if _has_kaggle_dirs():
        assert is_kaggle() is True
    else:
        assert is_kaggle() is False
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert is_kaggle() is True


def test_is_kaggle_tracks_competition_rerun_flag(monkeypatch):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "true")
    assert kaggle_env()["is_competition_rerun"] is True
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    assert kaggle_env()["is_competition_rerun"] is False


def test_kaggle_env_contract():
    env = kaggle_env()
    for key in (
        "is_kaggle",
        "kernel_run_type",
        "is_competition_rerun",
        "input_dir",
        "working_dir",
        "ram_gb",
        "cpu_flags",
        "cuda_devices",
        "has_internet",
    ):
        assert key in env
    assert isinstance(env["is_kaggle"], bool)
    assert env["kernel_run_type"] is None or isinstance(env["kernel_run_type"], str)
    assert env["ram_gb"] is None or env["ram_gb"] > 0
    assert isinstance(env["cpu_flags"], list)
    assert all(isinstance(f, str) for f in env["cpu_flags"])
    assert isinstance(env["cuda_devices"], list)
    assert isinstance(env["has_internet"], bool)


def test_cpuinfo_flags_never_crash_off_linux():
    feats = detect_cpu_features()
    flags = feats.get("cpu_flags", [])
    assert isinstance(flags, list)


def test_cuda_summary_contract():
    devices = cuda_device_summary()
    for dev in devices:
        assert "name" in dev and "memory_mb" in dev
        assert isinstance(dev["name"], str)
        assert isinstance(dev["memory_mb"], str)


def test_has_internet_is_boolean_and_offline_safe():
    assert has_internet(timeout=0.2) in (True, False)


def test_offline_sandbox_forward(monkeypatch):
    import socket

    def _blocked(*args, **kwargs):
        raise OSError("offline sandbox")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    assert has_internet() is False
    rng = __import__("numpy").random.default_rng(0)
    model = FeatherV1Model(make_config(dim=64, seq_len=64, num_chunks=2))
    out = model.forward(rng.standard_normal((8, 64)))
    assert "drafts" in out
    assert out["final_output"].shape == (1, 64)


def test_detect_cpu_features_contract():
    feats = detect_cpu_features()
    for key in (
        "avx512",
        "amx",
        "avx2",
        "avx",
        "neon",
        "machine",
        "cores_physical",
        "cores_logical",
        "cpu",
        "kaggle",
        "ram_gb",
    ):
        assert key in feats
    assert isinstance(feats["avx512"], bool)
    assert isinstance(feats["kaggle"], bool)
    assert feats["cores_physical"] >= 1
    assert feats["cores_logical"] >= feats["cores_physical"]


def test_get_best_kernel_contract():
    kernel = get_best_kernel()
    assert kernel["hypervector_dim"] in {512, 1024, 4096, 10000}
    assert kernel["binding"] in (
        "avx512_wht",
        "avx2_wht",
        "avx_wht",
        "neon_wht",
        "scalar_wht",
    )
    assert int(kernel["threads"]) >= 1
    assert "tok/s" in kernel["expected_tok_per_sec"]


def test_kernel_adaptive_to_detected_flags():
    feats = detect_cpu_features()
    kernel = get_best_kernel(feats)
    if feats["avx512"] and feats["amx"]:
        assert kernel["binding"] == "avx512_wht"
    elif feats["avx2"] and not (feats["avx512"] and feats["amx"]):
        assert kernel["binding"] == "avx2_wht"
        assert kernel["hypervector_dim"] >= 4096


def test_five_kaggle_configs_validate():
    assert len(KAGGLE_CONFIGS) == 5
    for path in KAGGLE_CONFIGS:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in REQUIRED_SKELETON_KEYS:
            assert key in data, f"{path.name} missing {key}"
        fv = data["feather_v1_config"]
        assert fv["ram_budget_gb"] > 0 and fv["ram_budget_gb"] <= 2.0
        assert fv["threads"] >= 1
        assert fv["hypervector_dim"] in {512, 1024, 4096, 10000}
        assert fv["binding"].lower() in {
            "auto",
            "avx512_wht",
            "avx2_wht",
            "avx_wht",
            "neon_wht",
            "scalar_wht",
        }
        assert "expected_performance" in data
        assert "tok_per_sec" in data["expected_performance"]


def test_kaggle_configs_load_through_api():
    for path in KAGGLE_CONFIGS:
        cfg = FeatherV1Config.from_file(str(path))
        assert cfg.ram_budget_gb > 0
        assert cfg.threads >= 1
        assert cfg.hypervector_dim in {512, 1024, 4096, 10000}
        assert cfg.vocab_size >= 256
        assert cfg.binding in (
            "avx512_wht",
            "avx2_wht",
            "avx_wht",
            "neon_wht",
            "scalar_wht",
        )
