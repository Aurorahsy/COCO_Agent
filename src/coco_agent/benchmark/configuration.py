"""Repository-external configuration for the local COCO_Benchmark adapter."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from ..configuration import config_path


BENCHMARK_CONFIG_FILENAME = "benchmark.json"


@dataclass(frozen=True)
class BenchmarkSettings:
    executable: str
    workdir: str
    output_dir: str
    api_key_env: str = "OPENAI_API_KEY"
    adapter: str = "coco_benchmark"


def benchmark_config_path(adapter: str = "coco_benchmark") -> Path:
    filename = BENCHMARK_CONFIG_FILENAME if adapter == "coco_benchmark" else f"benchmark-{adapter}.json"
    return config_path().with_name(filename)


def save_benchmark_settings(settings: BenchmarkSettings) -> Path:
    path = benchmark_config_path(settings.adapter)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, **asdict(settings)}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_benchmark_settings(adapter: str = "coco_benchmark") -> BenchmarkSettings | None:
    path = benchmark_config_path(adapter)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("adapter") != adapter:
        raise RuntimeError(f"Benchmark 配置格式无效：{path}")
    return BenchmarkSettings(
        executable=payload["executable"],
        workdir=payload["workdir"],
        output_dir=payload["output_dir"],
        api_key_env=payload.get("api_key_env", "OPENAI_API_KEY"),
        adapter=adapter,
    )
