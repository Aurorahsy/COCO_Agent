"""Stateful tuning intake tools exposed to the LLM."""

from __future__ import annotations

from copy import deepcopy
import re
from uuid import uuid4

from .tools import RegisteredTool, ToolRegistry


METRICS = (
    "throughput_tokens_per_second",
    "throughput_requests_per_second",
    "concurrency",
    "ttft",
    "tpot",
    "itl",
    "e2e",
    "success_rate",
    "goodput",
)


class TuningToolset:
    def __init__(self, benchmark_client=None) -> None:
        self._benchmark = benchmark_client
        self._tasks: dict[str, dict] = {}

    def update(self, arguments):
        task_id = arguments.get("task_id") or f"tuning-{uuid4().hex[:12]}"
        task = self._tasks.setdefault(
            task_id,
            {"task_id": task_id, "objective": {}, "constraints": [], "workload": {}, "target": {}, "benchmark": {}},
        )
        for section in ("objective", "workload", "target", "benchmark"):
            if arguments.get(section):
                task[section].update(arguments[section])
        self._normalize_target(task["target"])
        self._merge_constraints(task, arguments.get("constraints", []))
        missing = self._missing(task)
        task["missing_fields"] = missing
        task["ready_for_benchmark"] = not missing
        workload_missing = self._missing_for_workload(task)
        task["missing_workload_fields"] = workload_missing
        task["ready_for_workload"] = not workload_missing
        return deepcopy(task)

    def generate_workload(self, arguments):
        task_id = arguments["task_id"]
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"unknown task_id: {task_id}")
        missing = self._missing_for_workload(task)
        if missing:
            return {
                "task_id": task_id, "status": "needs_workload_context",
                "execution_state": "not_started", "missing_fields": missing,
            }
        if self._benchmark is None:
            return {
                "task_id": task_id, "status": "adapter_unconfigured",
                "execution_state": "not_started",
                "configuration_command": "coco benchmark config --adapter coco_benchmark",
            }
        result = self._benchmark.generate_workload(task_id, deepcopy(task))
        if result.get("status") == "workload_ready":
            task["workload"]["workload_ref"] = result["workload_ref"]
        return result

    def inspect_benchmarks(self, _arguments):
        if self._benchmark is None:
            return {"adapters": {}, "catalog": []}
        return self._benchmark.inspect()

    def prepare_benchmark(self, arguments):
        task_id = arguments["task_id"]
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"unknown task_id: {task_id}")
        missing = self._missing(task)
        if missing:
            return {"task_id": task_id, "status": "needs_context", "missing_fields": missing, "ready_for_benchmark": False}
        if self._benchmark is None:
            return {
                "task_id": task_id,
                "status": "benchmark_plan_ready",
                "execution_state": "adapter_unconfigured",
                "adapter": "coco_benchmark",
                "configuration_command": "coco benchmark config --adapter coco_benchmark",
                "ready_for_benchmark": True,
                "experiment_spec": deepcopy(task),
                "ui_event": {
                    "kind": "benchmark_plan",
                    "task_id": task_id,
                    "state": "adapter_unconfigured",
                    "adapter": "COCO_Benchmark",
                    "message": "COCO_Benchmark 尚未配置，请运行 coco benchmark config --adapter coco_benchmark",
                },
            }
        return self._benchmark.prepare(task_id, deepcopy(task))

    def confirm_benchmark(self, arguments):
        if self._benchmark is None:
            raise ValueError("Benchmark 执行适配器尚未配置")
        return self._benchmark.confirm(arguments["task_id"])

    def cancel_benchmark(self, arguments):
        if self._benchmark is None:
            raise ValueError("Benchmark 执行适配器尚未配置")
        return self._benchmark.cancel(arguments["task_id"])

    @staticmethod
    def _merge_constraints(task: dict, incoming: list[dict]) -> None:
        indexed = {
            (item["metric"], item.get("aggregation")): dict(item)
            for item in task["constraints"]
        }
        for item in incoming:
            metric, aggregation = item["metric"], item.get("aggregation")
            if aggregation:
                indexed.pop((metric, None), None)
            indexed[(metric, aggregation)] = dict(item)
        task["constraints"] = list(indexed.values())

    @staticmethod
    def _normalize_target(target: dict) -> None:
        endpoint = target.get("endpoint")
        if not isinstance(endpoint, str):
            return
        markdown_link = re.fullmatch(r"\[(https?://[^]]+)]\((https?://[^)]+)\)", endpoint.strip())
        if markdown_link:
            target["endpoint"] = markdown_link.group(2)

    @staticmethod
    def _missing_for_workload(task: dict) -> list[str]:
        missing: list[str] = []
        workload, target = task["workload"], task["target"]
        for field in ("input_tokens_min", "input_tokens_max", "output_tokens_max"):
            if workload.get(field) is None:
                missing.append(f"workload.{field}")
        if not target.get("model_ref"):
            missing.append("target.model_ref")
        return missing

    @staticmethod
    def _missing(task: dict) -> list[str]:
        missing: list[str] = []
        objective, workload, target = task["objective"], task["workload"], task["target"]
        if not objective.get("primary_metric"):
            missing.append("objective.primary_metric")
        if objective.get("target_value") is None:
            missing.append("objective.target_value")
        for index, constraint in enumerate(task["constraints"]):
            if not constraint.get("aggregation"):
                missing.append(f"constraints[{index}].aggregation")
        for field in ("input_tokens_min", "input_tokens_max"):
            if workload.get(field) is None:
                missing.append(f"workload.{field}")
        for field in (
            "model_ref", "engine_name", "engine_version", "deployment_ref", "endpoint",
            "launch_config_ref", "accelerator_model", "accelerator_count",
        ):
            if not target.get(field):
                missing.append(f"target.{field}")
        if target.get("benchmark_authorized") is not True:
            missing.append("target.benchmark_authorized")
        return missing

    def registry(self) -> ToolRegistry:
        metric_schema = {"type": "string", "enum": list(METRICS)}
        update_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "task_id": {"type": "string"},
                "objective": {"type": "object", "additionalProperties": False, "properties": {
                    "description": {"type": "string"}, "primary_metric": metric_schema,
                    "direction": {"type": "string", "enum": ["maximize", "minimize", "meet_target"]},
                    "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "=="]},
                    "target_value": {"type": "number"}, "unit": {"type": "string"},
                }},
                "constraints": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "properties": {"metric": metric_schema,
                        "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "=="]},
                        "value": {"type": "number"}, "unit": {"type": "string"},
                        "aggregation": {"type": "string", "enum": ["avg", "p50", "p95", "p99", "max"]}},
                    "required": ["metric", "operator", "value", "unit"]}},
                "workload": {"type": "object", "additionalProperties": False, "properties": {
                    "workload_ref": {"type": "string"},
                    "input_tokens_min": {"type": "integer", "minimum": 0},
                    "input_tokens_max": {"type": "integer", "minimum": 1},
                    "output_tokens_min": {"type": "integer", "minimum": 0},
                    "output_tokens_max": {"type": "integer", "minimum": 1},
                    "arrival_pattern": {"type": "string"}, "request_chain": {"type": "boolean"},
                    "content_kinds": {"type": "array", "items": {"type": "string"}},
                    "seed": {"type": "integer"},
                    "request_count": {"type": "integer", "minimum": 1},
                    "chains": {"type": "integer", "minimum": 1},
                    "turns_per_chain": {"type": "integer", "minimum": 1},
                    "growth_factor": {"type": "number", "exclusiveMinimum": 0},
                    "length_jitter": {"type": "number", "minimum": 0},
                    "chain_starts_per_second": {"type": "number", "exclusiveMinimum": 0},
                    "think_time_median_ms": {"type": "number", "minimum": 0},
                    "think_time_sigma": {"type": "number", "minimum": 0},
                }},
                "target": {"type": "object", "additionalProperties": False, "properties": {
                    "model_ref": {"type": "string"}, "engine_name": {"type": "string"},
                    "engine_version": {"type": "string"}, "deployment_ref": {"type": "string"},
                    "endpoint": {"type": "string"}, "launch_config_ref": {"type": "string"},
                    "accelerator_model": {"type": "string"},
                    "accelerator_count": {"type": "integer", "minimum": 1},
                    "benchmark_authorized": {"type": "boolean"},
                    "tokenizer_ref": {"type": "string"},
                    "tokenizer_revision": {"type": "string"},
                }},
                "benchmark": {"type": "object", "additionalProperties": False, "properties": {
                    "preferred_adapter": {
                        "type": "string", "enum": ["auto", "coco_benchmark", "ais_bench"]
                    },
                    "ais_model_profile": {"type": "string"},
                    "ais_dataset_profile": {"type": "string"},
                }},
            },
        }
        run_schema = {"type": "object", "additionalProperties": False, "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}
        empty_schema = {"type": "object", "additionalProperties": False, "properties": {}}
        return ToolRegistry([
            RegisteredTool("inspect_benchmark_capabilities", "Read the configured Benchmark versions, CLI options and documentation fingerprints. Use this instead of guessing available features.", empty_schema, self.inspect_benchmarks),
            RegisteredTool("update_tuning_task", "Create or update objectives, SLA constraints, workload, model, engine and deployment facts.", update_schema, self.update),
            RegisteredTool("generate_benchmark_workload", "Generate and persist a Benchmark workload independently of endpoint, credentials, deployment configuration and traffic authorization.", run_schema, self.generate_workload),
            RegisteredTool("prepare_benchmark_run", "Prepare a visible Benchmark execution plan after required context is complete. This does not mean the run has started.", run_schema, self.prepare_benchmark),
            RegisteredTool("confirm_benchmark_run", "Execute the prepared Benchmark plan after the user explicitly confirms this task. Never call without explicit confirmation.", run_schema, self.confirm_benchmark),
            RegisteredTool("cancel_benchmark_run", "Cancel a prepared Benchmark plan when the user explicitly declines or cancels it.", run_schema, self.cancel_benchmark),
        ])
