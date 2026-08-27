"""Single, repository-external LLM configuration with Windows DPAPI protection."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import tempfile
from ctypes import wintypes
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


APP_DIRECTORY = "coco_agent"
LEGACY_APP_DIRECTORY = "DeployOpt Agent"
CONFIG_FILENAME = "config.json"
CRYPTPROTECT_UI_FORBIDDEN = 0x1


def config_path() -> Path:
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return root / APP_DIRECTORY / CONFIG_FILENAME


def legacy_config_path() -> Path:
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return root / LEGACY_APP_DIRECTORY / CONFIG_FILENAME


def migrate_legacy_config() -> Path:
    """Move the former single config to the canonical coco_agent location."""
    target = config_path()
    legacy = legacy_config_path()
    if target.is_file() or not legacy.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    encrypted_config = legacy.read_bytes()
    try:
        with target.open("xb") as handle:
            handle.write(encrypted_config)
            handle.flush()
            os.fsync(handle.fileno())
        if target.read_bytes() != encrypted_config:
            raise RuntimeError("配置迁移完整性校验失败")
        try:
            legacy.unlink()
        except OSError:
            target.unlink(missing_ok=True)
            raise
    except Exception:
        target.unlink(missing_ok=True)
        raise
    try:
        legacy.parent.rmdir()
    except OSError:
        pass
    return target


def is_configured() -> bool:
    return migrate_legacy_config().is_file()


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(data: bytes):
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _windows_crypto():
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def _protect(secret: str) -> str:
    if os.name != "nt":
        raise RuntimeError("持久化密钥目前仅支持 Windows DPAPI。")
    crypt32, kernel32 = _windows_crypto()
    source, source_buffer = _blob(secret.encode("utf-8"))
    output = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(source),
        "coco_agent",
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output),
    ):
        raise ctypes.WinError()
    try:
        encrypted = ctypes.string_at(output.pbData, output.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        kernel32.LocalFree(output.pbData)
        del source_buffer


def _unprotect(encrypted: str) -> str:
    if os.name != "nt":
        raise RuntimeError("持久化密钥目前仅支持 Windows DPAPI。")
    crypt32, kernel32 = _windows_crypto()
    source, source_buffer = _blob(base64.b64decode(encrypted))
    output = _DataBlob()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output.pbData, output.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(output.pbData)
        del source_buffer


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    api_key: str


def save_settings(settings: LLMSettings) -> Path:
    if not settings.base_url or not settings.model or not settings.api_key:
        raise ValueError("base_url、model 和 api_key 均不能为空")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "provider": "openai_compatible",
        "base_url": settings.base_url.rstrip("/"),
        "model": settings.model,
        "api_key_dpapi": _protect(settings.api_key),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    # One canonical file only: atomic replacement, with no history or backup containing secrets.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="config-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def load_settings() -> LLMSettings:
    path = migrate_legacy_config()
    if not path.is_file():
        raise RuntimeError(f"尚未配置模型服务。请运行 coco-agent config。配置文件：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or "api_key_dpapi" not in payload:
        raise RuntimeError(f"配置文件格式无效：{path}")
    return LLMSettings(
        base_url=payload["base_url"],
        model=payload["model"],
        api_key=_unprotect(payload["api_key_dpapi"]),
    )


def public_settings() -> dict[str, str | bool]:
    path = migrate_legacy_config()
    if not path.is_file():
        return {"configured": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "configured": True,
        "path": str(path),
        "provider": payload.get("provider", "unknown"),
        "base_url": payload.get("base_url", ""),
        "model": payload.get("model", ""),
        "api_key": "已使用 Windows DPAPI 加密（不显示真值）",
        "updated_at": payload.get("updated_at", ""),
    }
