"""JSON-serializable execution projection for TuningGraph."""

from __future__ import annotations

from typing import Any, TypedDict


class TuningGraphState(TypedDict, total=False):
    task_id: str
    goal: dict[str, Any]
    status: str
    environment: dict[str, Any]
    experience_matches: list[dict[str, Any]]
    baseline_action: dict[str, Any]
    baseline_result: dict[str, Any]
    metric_analysis: dict[str, Any]
    recommendation: dict[str, Any]
    candidate_action: dict[str, Any]
    candidate_result: dict[str, Any]
    comparison_report: dict[str, Any]
    experiment_ids: list[str]
    pending_action: dict[str, Any]
    authorization: dict[str, Any]
    action_result: dict[str, Any]
    verification: dict[str, Any]
    experiment_id: str
    terminal_result: dict[str, Any]
