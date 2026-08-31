"""Benchmark adapter discovery, preference and capability comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkCapabilities:
    adapter: str
    display_name: str
    priority: int
    metrics: tuple[str, ...]
    aggregations: tuple[str, ...]
    workload_features: tuple[str, ...]
    evidence_features: tuple[str, ...]
    derived_metrics: tuple[str, ...]
    configuration_command: str


CATALOG = {
    "coco_benchmark": BenchmarkCapabilities(
        adapter="coco_benchmark",
        display_name="COCO_Benchmark",
        priority=100,
        metrics=("throughput_tokens_per_second", "throughput_requests_per_second", "ttft", "tpot", "itl", "e2e", "sla_goodput"),
        aggregations=("avg", "p50", "p95", "p99", "max"),
        workload_features=(
            "progressive_request_chains", "structured_content", "tool_schema",
            "arrival_trace", "workload_hash",
        ),
        evidence_features=("run_receipt", "request_events", "artifact_hashes", "client_self_check"),
        derived_metrics=("tpot=(e2e_ms-ttft_ms)/(output_tokens-1)",),
        configuration_command="coco benchmark config --adapter coco_benchmark",
    ),
    "ais_bench": BenchmarkCapabilities(
        adapter="ais_bench",
        display_name="AISBench",
        priority=80,
        metrics=("throughput_requests_per_second", "throughput_tokens_per_second", "ttft", "tpot", "itl", "e2e"),
        aggregations=("avg", "p50", "p75", "p90", "p99", "max", "min"),
        workload_features=("synthetic_dataset", "steady_state", "traffic_distribution", "accuracy_dataset"),
        evidence_features=("performance_report", "accuracy_report", "prediction_outputs"),
        derived_metrics=(),
        configuration_command="coco benchmark config --adapter ais_bench",
    ),
}


class BenchmarkAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, object] = {}
        self._pending: dict[str, tuple[object, dict]] = {}

    def register(self, adapter) -> None:
        self._adapters[adapter.name] = adapter

    def inspect(self) -> dict:
        snapshots = {}
        for name, adapter in self._adapters.items():
            inspector = getattr(adapter, "inspect", None)
            if inspector is None:
                continue
            try:
                snapshots[name] = inspector()
            except (OSError, RuntimeError, ValueError) as exc:
                snapshots[name] = {"adapter": name, "available": False, "error": str(exc)}
        return {"adapters": snapshots, "catalog": self.catalog()}

    def prepare(self, task_id: str, task: dict) -> dict:
        requested = task.get("benchmark", {}).get("preferred_adapter", "auto")
        selected = self._select(requested, task)
        if selected is None:
            preferred = "coco_benchmark" if requested == "auto" else requested
            descriptor = CATALOG[preferred]
            return {
                "task_id": task_id,
                "status": "benchmark_plan_ready",
                "execution_state": "adapter_unconfigured",
                "adapter": preferred,
                "configuration_command": descriptor.configuration_command,
                "available_adapters": self.catalog(),
                "ui_event": {
                    "kind": "benchmark_plan", "task_id": task_id,
                    "state": "adapter_unconfigured", "adapter": descriptor.display_name,
                    "message": f"{descriptor.display_name} 尚未配置，请运行 {descriptor.configuration_command}",
                },
            }
        result = selected.prepare(task_id, task)
        result["capability_match"] = self._capability_match(selected.name, task)
        result["available_adapters"] = self.catalog()
        if result.get("execution_state") == "awaiting_confirmation":
            self._pending[task_id] = (selected, result["plan"])
        return result

    def confirm(self, task_id: str) -> dict:
        pending = self._pending.get(task_id)
        if pending is None:
            raise ValueError(f"没有等待确认的 Benchmark 计划：{task_id}")
        adapter, plan = pending
        if plan.get("workload_generation_required"):
            return {
                "task_id": task_id,
                "status": "needs_workload",
                "execution_state": "not_started",
                "adapter": adapter.name,
                "missing_fields": ["workload.workload_ref"],
                "message": "执行尚未启动：请先生成或指定 workload_ref，然后重新生成计划",
                "ui_event": {
                    "kind": "benchmark_execution", "task_id": task_id,
                    "state": "needs_workload", "adapter": CATALOG[adapter.name].display_name,
                    "message": "执行尚未启动，需要先生成或指定 workload",
                },
            }
        try:
            result = adapter.execute(task_id, plan)
        except RuntimeError as exc:
            return {
                "task_id": task_id,
                "status": "execution_blocked",
                "execution_state": "not_started",
                "adapter": adapter.name,
                "message": str(exc),
                "ui_event": {
                    "kind": "benchmark_execution", "task_id": task_id,
                    "state": "blocked", "adapter": CATALOG[adapter.name].display_name,
                    "message": str(exc),
                },
            }
        if result.get("execution_state") in {"completed", "failed"}:
            self._pending.pop(task_id, None)
        return result

    def generate_workload(self, task_id: str, task: dict) -> dict:
        requested = task.get("benchmark", {}).get("preferred_adapter", "auto")
        name = "coco_benchmark" if requested == "auto" else requested
        adapter = self._adapters.get(name)
        if adapter is None:
            descriptor = CATALOG[name]
            return {
                "task_id": task_id,
                "status": "adapter_unconfigured",
                "execution_state": "not_started",
                "configuration_command": descriptor.configuration_command,
            }
        generator = getattr(adapter, "generate_workload", None)
        if generator is None:
            return {
                "task_id": task_id,
                "status": "workload_generation_unsupported",
                "execution_state": "not_started",
                "adapter": name,
            }
        return generator(task_id, task)

    def cancel(self, task_id: str) -> dict:
        if self._pending.pop(task_id, None) is None:
            raise ValueError(f"没有等待确认的 Benchmark 计划：{task_id}")
        return {"task_id": task_id, "status": "cancelled", "execution_state": "cancelled"}

    def _select(self, requested: str, task: dict):
        if requested != "auto":
            return self._adapters.get(requested)
        configured = [
            (self._score(name, task), adapter)
            for name, adapter in self._adapters.items()
            if name in CATALOG
        ]
        return max(configured, default=(0, None), key=lambda item: item[0])[1]

    def _score(self, name: str, task: dict | None) -> int:
        profile = CATALOG[name]
        if task is None:
            return profile.priority
        match = self._capability_match(name, task)
        return profile.priority + 100 * len(match["supported_requirements"])

    @staticmethod
    def _capability_match(name: str, task: dict) -> dict:
        profile = CATALOG[name]
        requirements = []
        primary = task.get("objective", {}).get("primary_metric")
        if primary:
            requirements.append((primary, None))
        requirements.extend(
            (item["metric"], item.get("aggregation"))
            for item in task.get("constraints", [])
        )
        supported, gaps = [], []
        for metric, aggregation in requirements:
            label = metric if not aggregation else f"{metric}:{aggregation}"
            if metric not in profile.metrics:
                gaps.append({"requirement": label, "reason": "metric_unavailable"})
            elif aggregation and aggregation not in profile.aggregations:
                gaps.append({"requirement": label, "reason": "aggregation_unavailable"})
            else:
                supported.append(label)
        return {"supported_requirements": supported, "gaps": gaps, "complete": not gaps}

    def catalog(self) -> list[dict]:
        return [
            asdict(profile) | {"configured": name in self._adapters}
            for name, profile in CATALOG.items()
        ]
