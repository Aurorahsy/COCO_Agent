"""Load native xxhash or install a pure-Python compatible module.

Some managed Windows hosts reject unsigned extension modules. LangGraph uses
XXH3-128 for deterministic internal identifiers, so the pure Python fallback
preserves the algorithm and output while trading only identifier-generation
speed.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def ensure_xxhash() -> None:
    if "xxhash" in sys.modules:
        return
    try:
        importlib.import_module("xxhash")
        return
    except ImportError:
        pass

    try:
        ppxxh = importlib.import_module("ppxxh")
    except ImportError as exc:
        raise RuntimeError(
            "xxhash native module could not be loaded and the ppxxh fallback is unavailable"
        ) from exc

    adapter = ModuleType("xxhash")
    adapter.__doc__ = "COCO_Agent pure-Python XXH3 compatibility module"
    adapter.xxh3_128 = ppxxh.xxh3_128

    def xxh3_128_hexdigest(data: bytes, seed: int = 0) -> str:
        return ppxxh.xxh3_128(data, seed=seed).hexdigest()

    adapter.xxh3_128_hexdigest = xxh3_128_hexdigest
    sys.modules["xxhash"] = adapter

