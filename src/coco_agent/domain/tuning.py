"""Domain contracts for evidence-driven inference tuning.

The types in this module are independent from LangGraph, concrete inference
engines, benchmark implementations, and telemetry vendors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Aggregation(str, Enum):
    AVG = "avg"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    MAX = "max"


class OptimizationDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class EvidenceAvailability(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class MetricConstraint:
    metric: str
    operator: str
    value: float
    unit: str
    aggregation: Aggregation


@dataclass(frozen=True)
class WorkloadIntent:
    workload_ref: str
    workload_hash: str | None = None
    arrival_pattern: str | None = None
    input_length_distribution: dict[str, float] = field(default_factory=dict)
    output_length_distribution: dict[str, float] = field(default_factory=dict)
    content_kinds: tuple[str, ...] = ()
    request_chain: bool = False


@dataclass(frozen=True)
class TargetIntent:
    model_ref: str
    deployment_ref: str


@dataclass(frozen=True)
class ExperimentBudget:
    max_runs: int = 1
    max_duration_seconds: float | None = None
    allow_profiling: bool = False
    allow_configuration_change: bool = False
    allow_restart: bool = False


@dataclass(frozen=True)
class PerformanceObjective:
    task_id: str
    primary_metric: str
    direction: OptimizationDirection
    workload: WorkloadIntent
    target: TargetIntent
    constraints: tuple[MetricConstraint, ...] = ()
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)


@dataclass(frozen=True)
class Provenance:
    source: str
    observed_at: str
    availability: EvidenceAvailability = EvidenceAvailability.AVAILABLE
    confidence: float = 1.0


@dataclass(frozen=True)
class ModelContext:
    model_ref: str
    revision: str | None = None
    tokenizer_ref: str | None = None
    architecture: str | None = None
    parameter_count: int | None = None
    layer_count: int | None = None
    hidden_size: int | None = None
    attention_heads: int | None = None
    kv_heads: int | None = None
    expert_count: int | None = None
    dtype: str | None = None
    quantization: str | None = None
    max_context_tokens: int | None = None
    weight_bytes: int | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class EngineContext:
    name: str
    version: str
    backend: str | None = None
    build: str | None = None
    effective_parameters: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None


@dataclass(frozen=True)
class DeploymentContext:
    kind: str
    endpoint_ref: str
    instance_count: int = 1
    launch_config_ref: str | None = None
    parallelism: dict[str, int] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None


@dataclass(frozen=True)
class AcceleratorDevice:
    vendor: str
    model: str
    count: int
    hbm_bytes_per_device: int | None = None
    hbm_bandwidth_bytes_per_second: float | None = None
    compute_flops_by_dtype: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareContext:
    accelerators: tuple[AcceleratorDevice, ...]
    interconnect: str | None = None
    interconnect_bandwidth_bytes_per_second: float | None = None
    topology: dict[str, Any] = field(default_factory=dict)
    host: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None


@dataclass(frozen=True)
class SoftwareComponent:
    name: str
    version: str
    build: str | None = None


@dataclass(frozen=True)
class MissingEvidence:
    field: str
    reason: str
    requested_capability: str | None = None


@dataclass(frozen=True)
class SystemContextSnapshot:
    snapshot_id: str
    captured_at: str
    model: ModelContext
    engine: EngineContext
    deployment: DeploymentContext
    hardware: HardwareContext
    software_stack: tuple[SoftwareComponent, ...] = ()
    missing_evidence: tuple[MissingEvidence, ...] = ()
    fingerprint: str | None = None


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    uri: str
    sha256: str | None = None


@dataclass(frozen=True)
class BenchmarkRunLink:
    phase: str
    run_id: str
    receipt_ref: str
    workload_hash: str
    started_at: str
    finished_at: str | None = None


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    objective: PerformanceObjective
    context: SystemContextSnapshot
    created_at: str
    benchmark_runs: tuple[BenchmarkRunLink, ...] = ()
    telemetry_artifacts: tuple[ArtifactRef, ...] = ()
    profiling_artifacts: tuple[ArtifactRef, ...] = ()
    configuration_changes: tuple[dict[str, Any], ...] = ()
    approvals: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TelemetryWindow:
    run_id: str
    started_at: str
    finished_at: str
    metrics: dict[str, float]
    artifacts: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class PerformanceEnvelope:
    weight_bytes: int | None
    estimated_kv_capacity_tokens: int | None
    compute_upper_bound_tokens_per_second: float | None
    memory_upper_bound_tokens_per_second: float | None
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class BottleneckHypothesis:
    category: str
    summary: str
    confidence: float
    evidence_refs: tuple[str, ...]
    counter_evidence_refs: tuple[str, ...] = ()
    missing_evidence: tuple[MissingEvidence, ...] = ()


@dataclass(frozen=True)
class EngineFeature:
    feature_id: str
    engine: str
    version_spec: str
    parameter: str
    description: str
    prerequisites: dict[str, Any]
    conflicts: tuple[str, ...]
    expected_effects: dict[str, str]
    risk: RiskLevel
    restart_required: bool
    validation_metrics: tuple[str, ...]
    source_url: str
    retrieved_at: str


@dataclass(frozen=True)
class OptimizationCandidate:
    candidate_id: str
    feature_id: str
    parameter: str
    previous_value: Any
    proposed_value: Any
    rationale: str
    target_metrics: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    risk: RiskLevel
    restart_required: bool
    rollback_value: Any


@dataclass(frozen=True)
class ComparisonSummary:
    experiment_id: str
    baseline_run_id: str
    candidate_run_id: str
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    absolute_delta: dict[str, float]
    relative_delta_percent: dict[str, float | None]
    constraint_results: dict[str, bool]
    conclusion: str
    evidence_refs: tuple[str, ...]
