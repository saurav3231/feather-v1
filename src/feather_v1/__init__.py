"""Feather v1 — CPU-native 200-year open source revolution LLM engine.

Six components, twelve advanced mathematics, hardware-adaptive kernels
(AVX-512 -> AVX2 -> AVX -> NEON -> Scalar). Version 1.0.0 Final.
"""

from .base import BaseComponent
from .config import FeatherV1Config
from .generation import GenerativeEvolution
from .governor import HomeostasisGovernor
from .hardware import (
    cuda_device_summary,
    detect_cpu_features,
    get_best_kernel,
    is_kaggle,
    kaggle_env,
)
from .knowledge import KnowledgeVault
from .memory import LiquidMemory
from .model import EnergyTracker, FeatherV1Model
from .reasoning import CognitiveWeaver
from .sensory import SensoryEncoder

__version__ = "1.0.0"
__all__ = [
    "FeatherV1Model",
    "FeatherV1Config",
    "BaseComponent",
    "SensoryEncoder",
    "LiquidMemory",
    "KnowledgeVault",
    "CognitiveWeaver",
    "HomeostasisGovernor",
    "GenerativeEvolution",
    "EnergyTracker",
    "get_best_kernel",
    "detect_cpu_features",
    "is_kaggle",
    "kaggle_env",
    "cuda_device_summary",
    "__version__",
]
