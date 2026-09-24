"""Feather v1 — adaptive hardware layer.

Detects the host CPU features and selects the best kernel configuration:

    AVX-512 + AMX  -> 10000-D hypervector (40KB L2), 8 binds/instruction, 94 tok/s
    AVX2           -> 4096-D hypervector (16KB L2), 4 binds/instruction, 45-60 tok/s
    AVX            -> 1024-D hypervector (4KB L1), 2 binds/instruction, 12-18 tok/s
    NEON           -> 1024-D hypervector (4KB L1), 35-50 tok/s (Apple M), 6 tok/s (Pi 5)
    Scalar         -> 512-D hypervector (2KB), 3-5 tok/s, works everywhere

The SIMD flags come from NumPy's ``__cpu_features__`` table (instant, no
subprocess).  py-cpuinfo is only used as optional brand-name enrichment and is
never allowed to block: it runs inside a daemon thread bounded by a timeout,
because its Windows subprocess probe can hang forever otherwise.

All components must import :func:`get_best_kernel` from here -- never
re-implement detection. Optimized for the Intel Core i5-3337U (Ivy Bridge,
2C/4T, 8GB, AVX but no AVX2/AVX-512/AMX), while working for all PCs.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import threading
from typing import Any

try:  # py-cpuinfo is optional; used only as a best-effort brand lookup.
    import cpuinfo  # type: ignore
except Exception:  # pragma: no cover - fallback detection
    cpuinfo = None  # type: ignore

_CPU_FEATURES = None
_CPU_INFO = None
_CPU_INFO_LOCK = threading.Lock()
_CUDA_DEVICES: list[dict[str, str]] | None = None


def _numpy_features() -> dict[str, bool]:
    """NumPy's CPU feature table: instant and subprocess-free."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feat_new

        feat = feat_new
    except Exception:  # pragma: no cover - older numpy versions
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as feat_last

            feat = feat_last
        except Exception:  # pragma: no cover
            return {}
    return {str(name): bool(value) for name, value in feat.items()}


def _features() -> dict[str, bool]:
    global _CPU_FEATURES
    if _CPU_FEATURES is None:
        _CPU_FEATURES = _numpy_features()
    return _CPU_FEATURES


def _flag(name: str, fallback: bool = False) -> bool:
    return bool(_features().get(name, fallback))


def _cpu_info() -> dict | None:
    """Bounded best-effort py-cpuinfo probe (never blocks the caller)."""
    if cpuinfo is None:
        return None
    global _CPU_INFO
    if _CPU_INFO is not None:
        return _CPU_INFO
    with _CPU_INFO_LOCK:
        if _CPU_INFO is not None:
            return _CPU_INFO
        result: dict = {}

        def _probe() -> None:
            try:
                result.update(cpuinfo.get_cpu_info())
            except Exception:  # pragma: no cover
                pass

        thread = threading.Thread(target=_probe, daemon=True)
        thread.start()
        thread.join(timeout=3.0)
        _CPU_INFO = result or None
    return _CPU_INFO


def _brand(info: dict | None) -> str:
    if info is not None:
        brand = info.get("brand_raw")
        if brand:
            return str(brand)
    return platform.processor() or "unknown"


def _arch_string(info: dict | None) -> str:
    if info is not None:
        arch = info.get("arch_string_raw")
        if arch:
            return str(arch)
    return platform.machine() or "unknown"


def _logical_cores() -> int:
    return os.cpu_count() or 1


def _physical_cores(info: dict | None) -> int:
    """Best guess: logical count halved when the probe did not answer."""
    logical = _logical_cores()
    if info is not None:
        try:
            count = int(info.get("count") or 0)
            if count > 0:
                return max(1, count // 2)
        except Exception:  # pragma: no cover
            pass
    return max(1, logical // 2)


def _proc_cpuinfo_flags() -> list[str]:
    """CPU flags from ``/proc/cpuinfo`` (empty on non-Linux hosts)."""
    path = "/proc/cpuinfo"
    if not os.path.exists(path):
        return []
    try:
        flags: list[str] = []
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip().lower()
                if line.startswith("flags") and ":" in line:
                    flags.extend(line.split(":", 1)[1].split())
        return flags
    except Exception:  # pragma: no cover - unreadable /proc
        return []


def _meminfo_gb() -> float | None:
    """Available RAM in GiB (Linux ``/proc/meminfo``, else psutil, else None)."""
    path = "/proc/meminfo"
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        kb = float(line.split()[1])
                        return kb / (1024.0 * 1024.0)
        except Exception:  # pragma: no cover - unreadable /proc
            pass
    try:  # psutil is optional; never required at import time.
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / (1024.0**3)
    except Exception:  # pragma: no cover - no psutil
        return None


def cuda_device_summary() -> list[dict[str, str]]:
    """NVIDIA GPUs via ``nvidia-smi`` (never imports torch/cuda runtime)."""
    global _CUDA_DEVICES
    if _CUDA_DEVICES is not None:
        return _CUDA_DEVICES
    try:
        import shutil

        exe = shutil.which("nvidia-smi")
    except Exception:  # pragma: no cover - import failure
        exe = None
    devices: list[dict[str, str]] = []
    if exe:
        try:
            out = subprocess.check_output(
                [
                    exe,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3,
            ).decode("utf-8", errors="ignore")
        except Exception:  # pragma: no cover - no GPU / no driver
            out = ""
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                devices.append({"name": parts[0], "memory_mb": parts[1]})
    _CUDA_DEVICES = devices
    return devices


def is_kaggle() -> bool:
    """True while running inside a Kaggle notebook/script runtime."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return os.path.exists("/kaggle/working") or os.path.exists("/kaggle/input")


def kaggle_env() -> dict[str, Any]:
    """Kaggle runtime facts: env vars, dirs, RAM, CPU flags, CUDA, network."""
    return {
        "is_kaggle": is_kaggle(),
        "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "is_competition_rerun": bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN")),
        "input_dir": "/kaggle/input" if os.path.isdir("/kaggle/input") else None,
        "working_dir": "/kaggle/working" if os.path.isdir("/kaggle/working") else None,
        "ram_gb": _meminfo_gb(),
        "cpu_flags": _proc_cpuinfo_flags(),
        "cuda_devices": cuda_device_summary(),
        "has_internet": has_internet(),
    }


def has_internet(timeout: float = 0.35) -> bool:
    """True if a short TCP probe reaches a public resolver (offline-safe)."""
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=timeout):
            return True
    except Exception:  # pragma: no cover - offline sandbox
        return False


def detect_cpu_features() -> dict[str, Any]:
    """Return detected CPU traits: SIMD flags, threads and machine type."""
    flags = _features()
    info = _cpu_info()
    cpu_flags = set(info.get("flags", [])) if info else set()
    machine = platform.machine().lower()
    return {
        "avx512": _flag("AVX512F") or "avx512f" in cpu_flags,
        "amx": any(name.startswith("AMX") for name in flags)
        or any(name.startswith("amx") for name in cpu_flags),
        "avx2": _flag("AVX2") or "avx2" in cpu_flags,
        "avx": _flag("AVX") or "avx" in cpu_flags,
        "neon": _flag("NEON") or machine in ("arm64", "aarch64"),
        "machine": machine,
        "cores_physical": _physical_cores(info),
        "cores_logical": _logical_cores(),
        "cpu": _brand(info),
        "arch_string_raw": _arch_string(info),
        "kaggle": is_kaggle(),
        "ram_gb": _meminfo_gb(),
    }


def get_best_kernel(features: dict[str, Any] | None = None) -> dict[str, Any]:
    """Choose the best kernel configuration for the host CPU.

    Implements the adaptive fallback chain:
    AVX-512 + AMX -> AVX2 -> AVX -> NEON -> Scalar.
    """
    if features is None:
        features = detect_cpu_features()

    avx512 = bool(features.get("avx512"))
    amx = bool(features.get("amx"))
    avx2 = bool(features.get("avx2"))
    avx = bool(features.get("avx"))
    neon = bool(features.get("neon"))
    physical = int(features.get("cores_physical") or 2)

    if avx512 and amx:
        return {
            "hypervector_dim": 10000,
            "hv_memory_kb": 40,
            "binding": "avx512_wht",
            "binds_per_instruction": 8,
            "moe": "amx_tropical_tt",
            "moe_tiles": "16x64",
            "precision": "bf16",
            "threads": physical,
            "cache": "L2",
            "expected_tok_per_sec": "94 tok/s",
        }
    if avx2:
        return {
            "hypervector_dim": 4096,
            "hv_memory_kb": 16,
            "binding": "avx2_wht",
            "binds_per_instruction": 4,
            "moe": "avx2_tropical_tt",
            "moe_tiles": "8x32",
            "precision": "int8",
            "threads": physical,
            "cache": "L2",
            "expected_tok_per_sec": "45-60 tok/s",
        }
    if avx:
        return {
            "hypervector_dim": 1024,
            "hv_memory_kb": 4,
            "binding": "avx_wht",
            "binds_per_instruction": 2,
            "moe": "avx_tropical_tt",
            "moe_tiles": "4x32",
            "precision": "int8",
            "threads": min(2, physical),
            "cache": "L1",
            "expected_tok_per_sec": "12-18 tok/s",
        }
    if neon:
        return {
            "hypervector_dim": 1024,
            "hv_memory_kb": 4,
            "binding": "neon_wht",
            "binds_per_instruction": 4,
            "moe": "neon_tropical",
            "moe_tiles": "4x16",
            "precision": "int8",
            "threads": physical,
            "cache": "L1",
            "expected_tok_per_sec": "35-50 tok/s (M3), 6 tok/s (Pi 5)",
        }
    return {
        "hypervector_dim": 512,
        "hv_memory_kb": 2,
        "binding": "scalar_wht",
        "binds_per_instruction": 1,
        "moe": "scalar_tropical",
        "moe_tiles": "1x1",
        "precision": "int8",
        "threads": 1,
        "cache": "L1",
        "expected_tok_per_sec": "3-5 tok/s",
    }


def load_config(path: str) -> dict:
    """Load a JSON config and merge in auto-detected hardware defaults."""
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    fv = cfg.setdefault("feather_v1_config", {})
    kernel = get_best_kernel()
    fv.setdefault("hypervector_dim", kernel["hypervector_dim"])
    fv.setdefault("binding", kernel["binding"])
    fv.setdefault("moe", kernel["moe"])
    fv.setdefault("threads", kernel["threads"])
    fv.setdefault("precision", kernel["precision"])
    fv["hardware_detected"] = detect_cpu_features()
    fv["kernel_selected"] = kernel
    return cfg


def summary() -> str:
    """Human-readable hardware summary (used by ``python -m feather_v1.hardware``)."""
    feats = detect_cpu_features()
    kernel = get_best_kernel(feats)
    lines = [
        f"CPU: {feats['cpu']}",
        f"Architecture: {feats['arch_string_raw']}",
        f"Cores: {feats['cores_physical']} physical / {feats['cores_logical']} logical",
        "Features: "
        + ", ".join(
            name
            for name, ok in [
                ("AVX-512", feats["avx512"]),
                ("AMX", feats["amx"]),
                ("AVX2", feats["avx2"]),
                ("AVX", feats["avx"]),
                ("NEON", feats["neon"]),
            ]
            if ok
        )
        or "Scalar (no SIMD)",
        "Best kernel: "
        + f"{kernel['binding'].upper()} binding, "
        + f"{kernel['hypervector_dim']}-D hypervector ({kernel['hv_memory_kb']}KB {kernel['cache']}), "
        + f"{kernel['moe']}, {kernel['threads']} threads, {kernel['precision']}",
        f"Expected: {kernel['expected_tok_per_sec']}",
    ]
    env = kaggle_env()
    if env["is_kaggle"]:
        lines.append(
            f"Kaggle runtime: {env['kernel_run_type'] or 'unknown'} "
            f"(competition rerun: {env['is_competition_rerun']})"
        )
    if env["ram_gb"] is not None:
        lines.append(f"RAM available: {env['ram_gb']:.1f} GB")
    if env["cuda_devices"]:
        lines.append(
            "CUDA: "
            + ", ".join(
                f"{dev['name']} {dev['memory_mb']}MB" for dev in env["cuda_devices"]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(summary())
