"""Feather v1 -- GGUF v3 reader/writer + converters.

    from feather_v1.gguf import (
        write_gguf, read_gguf, inspect_gguf, npz_round_trip,
        f16_pack, q8_0_pack, q4_k_pack,
    )

Header is exactly 24 bytes (magic 4 + version 4 + tensor_count 8 +
metadata_kv_count 8); tensor data aligned 32; ``gqa``-free, format v3.
"""

from .convert import convert_pt_to_gguf
from .reader import (
    GGUFReadResult,
    GGUFTensor,
    inspect_gguf,
    npz_round_trip,
    read_gguf,
)
from .real import REAL_METADATA_KEYS, build_real_gguf
from .writer import (
    QUANTIZERS,
    PackedTensor,
    f16_pack,
    f32_pack,
    q4_k_pack,
    q8_0_pack,
    write_gguf,
)

__all__ = [
    "GGUFReadResult",
    "GGUFTensor",
    "PackedTensor",
    "QUANTIZERS",
    "REAL_METADATA_KEYS",
    "build_real_gguf",
    "convert_pt_to_gguf",
    "f16_pack",
    "f32_pack",
    "inspect_gguf",
    "npz_round_trip",
    "q4_k_pack",
    "q8_0_pack",
    "read_gguf",
    "write_gguf",
]
