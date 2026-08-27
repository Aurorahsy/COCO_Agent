"""Stable, framework-independent contracts for the v4 implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationVerdict(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class AuthorizationVerdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class Criterion:
    metric: str
    operator: str
    target: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalSpec:
    task_id: str
    objective: str
    acceptance_criteria: tuple[Criterion, ...]
    inputs: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "acceptance_criteria": [item.to_dict() for item in self.acceptance_criteria],
            "inputs": dict(self.inputs),
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True)
class ActionRequest:
    capability: str
    arguments: dict[str, Any]
    operation_id: str
    idempotency_key: str
    risk: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    intent: str
    rationale: str
    action: ActionRequest
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "rationale": self.rationale,
            "action": self.action.to_dict(),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class AuthorizationDecision:
    verdict: AuthorizationVerdict
    reason: str
    policy_version: str = "poc-v1"


@dataclass(frozen=True)
class ActionResult:
    operation_id: str
    success: bool
    output: dict[str, Any]
    reused: bool = False
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Verification:
    verdict: VerificationVerdict
    evidence: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "evidence": dict(self.evidence),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MetricAnalysis:
    bottleneck: str
    evidence: dict[str, Any]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterRecommendation:
    parameter: str
    previous_value: Any
    recommended_value: Any
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonReport:
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    absolute_delta: dict[str, float]
    relative_delta_pct: dict[str, float | None]
    recommendation: dict[str, Any]
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
