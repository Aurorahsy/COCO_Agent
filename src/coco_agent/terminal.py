"""Small terminal renderer for the interactive coco experience."""

from __future__ import annotations

import itertools
import json
import sys
import threading
from collections.abc import Callable


class Spinner:
    """Render a transient spinner while a blocking model request is running."""

    def __init__(self, label: str = "coco 正在思考") -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if not sys.stdout.isatty():
            return self
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def _animate(self) -> None:
        frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        while not self._stop.wait(0.08):
            sys.stdout.write(f"\r  {next(frames)} {self._label}…")
            sys.stdout.flush()

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()


class TerminalRenderer:
    def __init__(self, output: Callable[[str], None], *, animate: bool) -> None:
        self._output = output
        self._animate = animate

    def prompt(self) -> str:
        if self._animate:
            return "\x1b[1;36m❯\x1b[0m "
        return "❯ "

    def thinking(self):
        return Spinner() if self._animate else _NoopContext()

    def events(self, events: list[tuple[str, str]]) -> None:
        for kind, text in events:
            if kind == "reasoning":
                self._trace(f"思考  {text}", branch="├─")
            elif kind == "tool":
                self._trace(f"工具  {text}", branch="├─")
            elif kind == "tool_result":
                self._trace(f"结果  {text}", branch="└─")
            elif kind == "ui_event":
                self._ui_event(text)

    def _ui_event(self, raw: str) -> None:
        event = json.loads(raw)
        if event.get("kind") not in {"benchmark_plan", "benchmark_execution", "workload_generation"}:
            return
        state_labels = {
            "adapter_unconfigured": "尚未启动 · 执行适配器待配置",
            "awaiting_confirmation": "等待用户确认",
            "running": "正在运行",
            "summarizing": "正在汇总",
            "completed": "已完成",
            "failed": "运行失败",
            "blocked": "尚未启动 · 执行条件不足",
            "needs_workload": "尚未启动 · 需要工作负载",
            "cancelled": "已取消",
        }
        self._output("◇ Benchmark")
        if event.get("adapter"):
            self._output(f"  适配器  {event['adapter']}")
        self._output(f"  任务  {event.get('task_id', '-')}")
        self._output(f"  状态  {state_labels.get(event.get('state'), event.get('state', '-'))}")
        if event.get("message"):
            self._output(f"  说明  {event['message']}")
        self._output("")

    def _trace(self, text: str, *, branch: str) -> None:
        line = f"  {branch} {text}"
        self._output(f"\x1b[2;90m{line}\x1b[0m" if self._animate else line)

    def response(self, text: str) -> None:
        label = "● coco"
        self._output(f"\x1b[1;32m{label}\x1b[0m" if self._animate else label)
        for line in (text or "").splitlines() or [""]:
            self._output(f"  {line}")
        self._output("")

    def error(self, text: str) -> None:
        label = "● coco"
        warning = f"  ⚠ {text}"
        if self._animate:
            self._output(f"\x1b[1;31m{label}\x1b[0m")
            self._output(f"\x1b[31m{warning}\x1b[0m")
        else:
            self._output(label)
            self._output(warning)
        self._output("")


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None
