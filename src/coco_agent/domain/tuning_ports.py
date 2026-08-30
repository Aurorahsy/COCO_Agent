"""Ports for the inference tuning modules.

Ports are owned by the Agent application/domain boundary. Concrete engine,
accelerator, benchmark, storage, and native implementations live in adapters.
"""

from __future__ import annotations

from typing import Any, Protocol

from .tuning import (
    ArtifactRef,
    BottleneckHypothesis,
    ComparisonSummary,
    DeploymentContext,
    EngineContext,
    EngineFeature,
    ExperimentManifest,
    HardwareContext,
    ModelContext,
    OptimizationCandidate,
    PerformanceEnvelope,
    PerformanceObjective,
    SystemContextSnapshot,
    TelemetryWindow,
)


class ModelInspector(Protocol):
    def inspect(self, model_ref: str) -> ModelContext: ...


class EngineInspector(Protocol):
    def inspect(self, target_ref: str) -> EngineContext: ...


class DeploymentInspector(Protocol):
    def inspect(self, target_ref: str) -> DeploymentContext: ...


class HardwareInspector(Protocol):
    def inspect(self, target_ref: str) -> HardwareContext: ...


class ContextCollector(Protocol):
    def collect(self, objective: PerformanceObjective) -> SystemContextSnapshot: ...


class ExperimentManifestRepository(Protocol):
    def save(self, manifest: ExperimentManifest) -> None: ...
    def get(self, experiment_id: str) -> ExperimentManifest | None: ...


class BenchmarkClient(Protocol):
    def submit(self, experiment_ref: str, spec: dict[str, Any]) -> str: ...
    def status(self, run_id: str) -> dict[str, Any]: ...
    def result(self, run_id: str) -> dict[str, Any]: ...
    def cancel(self, run_id: str) -> None: ...


class TelemetryCollector(Protocol):
    def collect(self, run_id: str, started_at: str, finished_at: str) -> TelemetryWindow: ...


class ProfilingController(Protocol):
    def capture(self, run_id: str, profile_spec: dict[str, Any]) -> ArtifactRef: ...


class PerformanceModel(Protocol):
    def estimate(self, context: SystemContextSnapshot) -> PerformanceEnvelope: ...


class BottleneckDiagnoser(Protocol):
    def diagnose(
        self,
        objective: PerformanceObjective,
        context: SystemContextSnapshot,
        metrics: dict[str, float],
        telemetry: TelemetryWindow | None,
        envelope: PerformanceEnvelope,
    ) -> tuple[BottleneckHypothesis, ...]: ...


class EngineKnowledgeProvider(Protocol):
    def find_features(
        self,
        context: SystemContextSnapshot,
        hypothesis: BottleneckHypothesis,
    ) -> tuple[EngineFeature, ...]: ...


class CandidatePlanner(Protocol):
    def plan(
        self,
        objective: PerformanceObjective,
        context: SystemContextSnapshot,
        hypotheses: tuple[BottleneckHypothesis, ...],
        features: tuple[EngineFeature, ...],
    ) -> tuple[OptimizationCandidate, ...]: ...


class DeploymentController(Protocol):
    def snapshot(self, target_ref: str) -> ArtifactRef: ...
    def validate(self, candidate: OptimizationCandidate) -> dict[str, Any]: ...
    def apply(self, candidate: OptimizationCandidate, idempotency_key: str) -> str: ...
    def restart(self, target_ref: str, idempotency_key: str) -> str: ...
    def health(self, target_ref: str) -> dict[str, Any]: ...
    def rollback(self, snapshot: ArtifactRef, idempotency_key: str) -> str: ...


class TuningReportBuilder(Protocol):
    def build(
        self,
        manifest: ExperimentManifest,
        hypotheses: tuple[BottleneckHypothesis, ...],
        candidate: OptimizationCandidate,
        baseline: dict[str, float],
        candidate_metrics: dict[str, float],
    ) -> ComparisonSummary: ...

