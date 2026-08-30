"""Execution-plan adapter dedicated to the independent COCO_Benchmark project."""

from __future__ import annotations

from pathlib import Path
import json
import hashlib
import math
import re
from urllib.parse import urlsplit

from .configuration import BenchmarkSettings
from .runner import SubprocessRunner, secret_environment, require_workload


class LocalCocoBenchmarkAdapter:
    name = "coco_benchmark"

    def __init__(
        self, settings: BenchmarkSettings, runner=None, credential: str | None = None
    ) -> None:
        self.settings = settings
        self._runner = runner or SubprocessRunner()
        self._credential = credential

    def prepare(self, task_id: str, task: dict) -> dict:
        executable = Path(self.settings.executable)
        workdir = Path(self.settings.workdir)
        target = task["target"]
        base_endpoint = self._base_endpoint(target["endpoint"])
        workload_ref = task["workload"].get("workload_ref")
        plan = {
            "adapter": self.name,
            "executable": str(executable),
            "workdir": str(workdir),
            "output_dir": self.settings.output_dir,
            "api_key_env": self.settings.api_key_env,
            "model": target["model_ref"],
            "endpoint": base_endpoint,
            "engine": self._engine_name(target["engine_name"]),
            "workload_ref": workload_ref,
            "workload_generation_required": not bool(workload_ref),
        }
        return {
            "task_id": task_id,
            "status": "benchmark_plan_ready",
            "execution_state": "awaiting_confirmation",
            "adapter": self.name,
            "plan": plan,
            "experiment_spec": task,
            "ui_event": {
                "kind": "benchmark_plan",
                "task_id": task_id,
                "state": "awaiting_confirmation",
                "adapter": "COCO_Benchmark",
                "message": "COCO_Benchmark 执行计划已生成，等待用户确认",
            },
        }

    def generate_workload(self, task_id: str, task: dict) -> dict:
        workload = task["workload"]
        target = task["target"]
        bundle_dir = Path(self.settings.output_dir) / "workloads" / task_id
        config_path = bundle_dir / "build-config.json"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_ref = target.get("tokenizer_ref")
        tokenization = (
            {
                "backend": "huggingface",
                "model": tokenizer_ref,
                **({"revision": target["tokenizer_revision"]} if target.get("tokenizer_revision") else {}),
            }
            if tokenizer_ref else {"backend": "simple"}
        )
        request_count = workload.get("request_count")
        if request_count:
            chains, turns = self._shape_for_request_count(
                int(request_count), workload.get("chains")
            )
        else:
            chains = int(workload.get("chains", 4))
            turns = int(workload.get("turns_per_chain", 4))
        config = {
            "schema_version": "1.0",
            "dataset": {
                "id": f"{task_id}-workload",
                "seed": int(workload.get("seed", 42)),
                "chains": chains,
                "turns_per_chain": turns,
                "max_output_tokens": int(workload.get("output_tokens_max", 256)),
                "tokenization": tokenization,
                "length": {
                    "start_tokens": int(workload["input_tokens_min"]),
                    "growth_factor": float(workload.get("growth_factor", 2.0)),
                    "max_tokens": int(workload["input_tokens_max"]),
                    "jitter": float(workload.get("length_jitter", 0.05)),
                },
                "timing": {
                    "mode": workload.get("arrival_pattern", "poisson"),
                    "chain_starts_per_second": float(workload.get("chain_starts_per_second", 1.0)),
                    "think_time_median_ms": float(workload.get("think_time_median_ms", 2500)),
                    "think_time_sigma": float(workload.get("think_time_sigma", 0.8)),
                },
            },
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        result = self._runner.run(
            [
                self.settings.executable, "build-dataset",
                "--config", str(config_path),
                "--output-dir", str(bundle_dir),
            ],
            cwd=self.settings.workdir,
        )
        requests_path = bundle_dir / "requests.jsonl"
        if result.returncode != 0 or not requests_path.is_file():
            return {
                "task_id": task_id,
                "status": "workload_generation_failed",
                "execution_state": "failed",
                "returncode": result.returncode,
                "stderr": result.stderr[-8000:],
            }
        return {
            "task_id": task_id,
            "status": "workload_ready",
            "execution_state": "completed",
            "workload_ref": str(requests_path),
            "manifest_ref": str(bundle_dir / "manifest.json"),
            "tokenizer_mode": tokenization["backend"],
            "request_count": chains * turns,
            "dataset_shape": {"chains": chains, "turns_per_chain": turns},
            "ui_event": {
                "kind": "workload_generation", "task_id": task_id,
                "state": "completed", "adapter": "COCO_Benchmark",
                "message": "工作负载已生成并通过落地检查",
            },
        }

    def inspect(self) -> dict:
        commands = {}
        for name, arguments in {
            "root": ["--help"],
            "build-dataset": ["build-dataset", "--help"],
            "run": ["run", "--help"],
        }.items():
            result = self._runner.run(
                [self.settings.executable, *arguments], cwd=self.settings.workdir
            )
            text = f"{result.stdout}\n{result.stderr}"
            commands[name] = {
                "available": result.returncode == 0,
                "options": sorted(set(re.findall(r"--[a-z0-9-]+", text))),
            }
        version_result = self._runner.run(
            [self.settings.executable, "--version"], cwd=self.settings.workdir
        )
        docs = Path(self.settings.workdir) / "docs" / "USAGE.md"
        docs_info = {"path": str(docs), "available": docs.is_file()}
        if docs.is_file():
            content = docs.read_bytes()
            docs_info["sha256"] = hashlib.sha256(content).hexdigest()
        return {
            "adapter": self.name,
            "display_name": "COCO_Benchmark",
            "version": (version_result.stdout or version_result.stderr).strip(),
            "commands": commands,
            "documentation": docs_info,
            "dataset_controls": {
                "request_count_compiles_to": "chains * turns_per_chain",
                "length": ["start_tokens", "growth_factor", "max_tokens", "jitter", "target_tokens"],
                "timing": ["chain_starts_per_second", "think_time_median_ms", "think_time_sigma"],
            },
        }

    @staticmethod
    def _shape_for_request_count(request_count: int, preferred_chains=None) -> tuple[int, int]:
        if request_count <= 0:
            raise ValueError("workload.request_count must be positive")
        if preferred_chains:
            chains = int(preferred_chains)
            if request_count % chains:
                raise ValueError("request_count 必须能被显式 chains 整除")
            return chains, request_count // chains
        for chains in range(math.isqrt(request_count), 0, -1):
            if request_count % chains == 0:
                return chains, request_count // chains
        return 1, request_count
    def execute(self, task_id: str, plan: dict) -> dict:
        environment = secret_environment(plan["api_key_env"], self._credential)
        workload = require_workload(plan.get("workload_ref"), cwd=plan["workdir"])
        command = [
            plan["executable"], "run",
            "--workload", workload,
            "--endpoint", plan["endpoint"],
            "--model", plan["model"],
            "--engine", plan["engine"],
            "--output-dir", plan["output_dir"],
            "--run-id", task_id,
            "--api-key-env", plan["api_key_env"],
            "--output-format", "json",
        ]
        result = self._runner.run(
            command, cwd=plan["workdir"], environment=environment
        )
        state = "completed" if result.returncode == 0 else "failed"
        return {
            "task_id": task_id,
            "status": "benchmark_completed" if state == "completed" else "benchmark_failed",
            "execution_state": state,
            "adapter": self.name,
            "run_ref": str(Path(plan["output_dir"]) / task_id),
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
            "ui_event": {
                "kind": "benchmark_execution", "task_id": task_id,
                "state": state, "adapter": "COCO_Benchmark",
                "message": "COCO_Benchmark 已完成" if state == "completed" else "COCO_Benchmark 运行失败",
            },
        }
    @staticmethod
    def _engine_name(value: str) -> str:
        normalized = value.casefold().replace("_", "-")
        return {"vllm": "vllm", "vllm-ascend": "vllm-ascend", "mindie": "mindie"}.get(
            normalized, normalized
        )

    @staticmethod
    def _base_endpoint(endpoint: str) -> str:
        parsed = urlsplit(endpoint)
        path = parsed.path.rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        return parsed._replace(path=path, query="", fragment="").geturl().rstrip("/")
