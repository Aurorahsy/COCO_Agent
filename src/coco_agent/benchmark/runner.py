"""Host process execution isolated from Benchmark selection and conversation policy."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


class SubprocessRunner:
    def run(
        self, command: list[str], *, cwd: str,
        environment: dict[str, str] | None = None,
    ) -> ProcessResult:
        process_environment = os.environ.copy()
        if environment:
            process_environment.update(environment)
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=process_environment,
        )
        return ProcessResult(completed.returncode, completed.stdout, completed.stderr)


def secret_environment(name: str, configured_secret: str | None = None) -> dict[str, str]:
    """Resolve a secret without returning it through a tool result or command line."""
    value = configured_secret or os.environ.get(name)
    if not name or not value:
        raise RuntimeError(
            f"目标服务凭据未配置：请运行 coco benchmark credential 或设置环境变量 {name}"
        )
    return {name: value}


def require_workload(path: str | None, *, cwd: str) -> str:
    if not path:
        raise RuntimeError("workload 尚未生成；请先生成或指定 workload_ref")
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(cwd) / resolved
    if not resolved.is_file():
        raise RuntimeError(f"workload 文件不存在：{resolved}")
    return str(resolved)
