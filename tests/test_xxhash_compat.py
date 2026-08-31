from __future__ import annotations

import sys
from types import SimpleNamespace

from coco_agent.compat import xxhash as compatibility


def test_pure_python_fallback_exposes_langgraph_api(monkeypatch):
    class Hash:
        def hexdigest(self):
            return "0123456789abcdef" * 2

        def digest(self):
            return bytes.fromhex(self.hexdigest())

    fake_ppxxh = SimpleNamespace(xxh3_128=lambda data, seed=0: Hash())
    real_import = compatibility.importlib.import_module

    def fake_import(name):
        if name == "xxhash":
            raise ImportError("native module blocked")
        if name == "ppxxh":
            return fake_ppxxh
        return real_import(name)

    original = sys.modules.pop("xxhash", None)
    try:
        monkeypatch.setattr(compatibility.importlib, "import_module", fake_import)
        compatibility.ensure_xxhash()
        import xxhash

        assert xxhash.xxh3_128_hexdigest(b"state") == "0123456789abcdef" * 2
        assert xxhash.xxh3_128(b"state").digest() == bytes.fromhex("0123456789abcdef" * 2)
    finally:
        sys.modules.pop("xxhash", None)
        if original is not None:
            sys.modules["xxhash"] = original
