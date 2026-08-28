"""High-level tuning tools available to the LLM interpreter."""

from __future__ import annotations

from uuid import uuid4

from ..domain.contracts import Criterion, GoalSpec
from .tools import RegisteredTool, ToolRegistry


class TuningToolset:
    def __init__(self, application) -> None:
        self._application = application
        self._goals: dict[str, GoalSpec] = {}

    def submit(self, arguments):
        task_id = f"llm-{uuid4().hex[:12]}"
        goal = GoalSpec(
            task_id=task_id,
            objective=arguments["objective"],
            acceptance_criteria=(Criterion(
                arguments["metric"], arguments["operator"], arguments["target"]
            ),),
            inputs={
                "hardware": "mock-gpu",
                "model": "mock-model",
                "baseline_config": {"max_num_seqs": arguments.get("max_num_seqs", 128)},
                "baseline_metrics": {"throughput": 80.0, "latency_p99_ms": 220.0},
                "candidate_metrics": {"throughput": 120.0, "latency_p99_ms": 170.0},
            },
        )
        self._goals[task_id] = goal
        return {"task_id": task_id, "goal": goal.to_dict(), "status": "accepted"}

    def run(self, arguments):
        task_id = arguments["task_id"]
        goal = self._goals.get(task_id)
        if goal is None:
            raise ValueError(f"unknown task_id: {task_id}")
        result = self._application.start(goal)
        return {
            "task_id": task_id,
            "status": result["status"],
            "environment": result.get("environment"),
            "analysis": result.get("metric_analysis"),
            "recommendation": result.get("recommendation"),
            "report": result.get("terminal_result", {}).get("comparison_report"),
        }

    def registry(self) -> ToolRegistry:
        submit_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "objective": {"type": "string"},
                "metric": {"type": "string", "enum": ["throughput"]},
                "operator": {"type": "string", "enum": [">=", ">", "=="]},
                "target": {"type": "number", "exclusiveMinimum": 0},
                "max_num_seqs": {"type": "integer", "exclusiveMinimum": 0},
            },
            "required": ["objective", "metric", "operator", "target"],
        }
        run_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        }
        return ToolRegistry([
            RegisteredTool(
                "submit_tuning_task",
                "Parse and submit a throughput tuning goal. Call only when target is explicit.",
                submit_schema,
                self.submit,
            ),
            RegisteredTool(
                "run_tuning_task",
                "Collect environment, benchmark, recommend one parameter, retest, and report.",
                run_schema,
                self.run,
            ),
        ])
