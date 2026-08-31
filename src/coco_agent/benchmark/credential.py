"""Private target-service credentials stored outside the source repository."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..configuration import config_path


def credential_path() -> Path:
    return config_path().with_name("benchmark-credentials.json")


def save_credential(adapter: str, api_key: str) -> Path:
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "credentials": {}}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("version") == 1:
            payload = existing
    payload.setdefault("credentials", {})[adapter] = api_key
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_credential(adapter: str) -> str | None:
    path = credential_path()
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise RuntimeError(f"Benchmark 凭据配置格式无效：{path}")
    value = payload.get("credentials", {}).get(adapter)
    return value if isinstance(value, str) and value else None
