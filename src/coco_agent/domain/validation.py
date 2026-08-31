"""Validation for tuning domain contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .tuning import PerformanceObjective, SystemContextSnapshot


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    code: str
    message: str


class ObjectiveValidator:
    _OPERATORS = {">", ">=", "<", "<=", "=="}

    def validate(self, objective: PerformanceObjective) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if not objective.task_id.strip():
            issues.append(ValidationIssue("task_id", "required", "task_id is required"))
        if not objective.primary_metric.strip():
            issues.append(
                ValidationIssue("primary_metric", "required", "primary metric is required")
            )
        if not objective.workload.workload_ref.strip():
            issues.append(
                ValidationIssue("workload.workload_ref", "required", "workload is required")
            )
        if not objective.target.model_ref.strip():
            issues.append(
                ValidationIssue("target.model_ref", "required", "model reference is required")
            )
        if not objective.target.deployment_ref.strip():
            issues.append(
                ValidationIssue(
                    "target.deployment_ref", "required", "deployment reference is required"
                )
            )
        if objective.budget.max_runs < 1:
            issues.append(
                ValidationIssue("budget.max_runs", "range", "max_runs must be positive")
            )
        for index, constraint in enumerate(objective.constraints):
            prefix = f"constraints[{index}]"
            if constraint.operator not in self._OPERATORS:
                issues.append(
                    ValidationIssue(f"{prefix}.operator", "unsupported", "unsupported operator")
                )
            if not constraint.unit.strip():
                issues.append(ValidationIssue(f"{prefix}.unit", "required", "unit is required"))
        return tuple(issues)


class ContextValidator:
    def validate(self, context: SystemContextSnapshot) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if not context.model.model_ref.strip():
            issues.append(ValidationIssue("model.model_ref", "required", "model is required"))
        if not context.engine.name.strip() or not context.engine.version.strip():
            issues.append(
                ValidationIssue("engine", "required", "engine name and version are required")
            )
        if not context.hardware.accelerators:
            issues.append(
                ValidationIssue("hardware.accelerators", "required", "accelerator is required")
            )
        return tuple(issues)
