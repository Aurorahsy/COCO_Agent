"""Execution-plan adapter for AISBench performance evaluation."""

from __future__ import annotations

from .configuration import BenchmarkSettings
from .runner import SubprocessRunner, secret_environment


class LocalAisBenchAdapter:
    name = "ais_bench"

    def __init__(
        self, settings: BenchmarkSettings, runner=None, credential: str | None = None
    ) -> None:
        self.settings = settings
        self._runner = runner or SubprocessRunner()
        self._credential = credential

    def prepare(self, task_id: str, task: dict) -> dict:
        benchmark = task.get("benchmark", {})
        model_profile = benchmark.get("ais_model_profile")
        dataset_profile = benchmark.get("ais_dataset_profile")
        missing = []
        if not model_profile:
            missing.append("benchmark.ais_model_profile")
        if not dataset_profile:
            missing.append("benchmark.ais_dataset_profile")
        if missing:
            return {
                "task_id": task_id,
                "status": "needs_adapter_context",
                "execution_state": "not_started",
                "adapter": self.name,
                "missing_fields": missing,
                "adapter_usage": "ais_bench --models MODEL_PROFILE --datasets DATASET_PROFILE --mode perf",
            }
        command = [
            self.settings.executable,
            "--models", model_profile,
            "--datasets", dataset_profile,
            "--mode", "perf",
        ]
        return {
            "task_id": task_id,
            "status": "benchmark_plan_ready",
            "execution_state": "awaiting_confirmation",
            "adapter": self.name,
            "plan": {
                "command": command,
                "workdir": self.settings.workdir,
                "output_dir": self.settings.output_dir,
                "api_key_env": self.settings.api_key_env,
            },
            "ui_event": {
                "kind": "benchmark_plan", "task_id": task_id,
                "state": "awaiting_confirmation", "adapter": "AISBench",
                "message": "AISBench perf 执行计划已生成，等待用户确认",
            },
        }

    def execute(self, task_id: str, plan: dict) -> dict:
        environment = secret_environment(plan["api_key_env"], self._credential)
        result = self._runner.run(
            plan["command"], cwd=plan["workdir"], environment=environment
        )
        state = "completed" if result.returncode == 0 else "failed"
        return {
            "task_id": task_id,
            "status": "benchmark_completed" if state == "completed" else "benchmark_failed",
            "execution_state": state,
            "adapter": self.name,
            "run_ref": plan["output_dir"],
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
            "ui_event": {
                "kind": "benchmark_execution", "task_id": task_id,
                "state": state, "adapter": "AISBench",
                "message": "AISBench 已完成" if state == "completed" else "AISBench 运行失败",
            },
        }

    def inspect(self) -> dict:
        result = self._runner.run(
            [self.settings.executable, "-h"], cwd=self.settings.workdir
        )
        return {
            "adapter": self.name,
            "display_name": "AISBench",
            "version": "unknown",
            "commands": {"root": {"available": result.returncode == 0}},
        }
