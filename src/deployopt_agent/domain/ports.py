"""Ports owned by the domain/application boundary."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import (
    ActionRequest,
    ActionResult,
    AuthorizationDecision,
    GoalSpec,
    MetricAnalysis,
    ParameterRecommendation,
    ComparisonReport,
    Verification,
)


class Capability(Protocol):
    def inspect_environment(self, goal: GoalSpec) -> dict[str, Any]: ...
    def execute(self, request: ActionRequest) -> ActionResult: ...
    def status(self, operation_id: str) -> ActionResult | None: ...


class AuthorizationService(Protocol):
    def authorize(self, request: ActionRequest) -> AuthorizationDecision: ...


class ExperienceService(Protocol):
    def retrieve(self, context: dict[str, Any]) -> list[dict[str, Any]]: ...
    def record_experiment(self, record: dict[str, Any]) -> str: ...


class GoalVerifier(Protocol):
    def verify(self, goal: GoalSpec, result: ActionResult) -> Verification: ...


class TuningPolicy(Protocol):
    def analyze(self, goal: GoalSpec, baseline_metrics: dict[str, float]) -> MetricAnalysis: ...
    def recommend(
        self, goal: GoalSpec, analysis: MetricAnalysis
    ) -> ParameterRecommendation: ...


class ReportBuilder(Protocol):
    def build(
        self,
        baseline: dict[str, float],
        candidate: dict[str, float],
        recommendation: ParameterRecommendation,
        verification: Verification,
    ) -> ComparisonReport: ...
