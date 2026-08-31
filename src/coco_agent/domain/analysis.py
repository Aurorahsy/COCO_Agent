"""Deterministic first-pass performance analysis and candidate planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tuning import (
    BottleneckHypothesis,
    ComparisonSummary,
    EngineFeature,
    MissingEvidence,
    OptimizationCandidate,
    PerformanceEnvelope,
    PerformanceObjective,
    RiskLevel,
    SystemContextSnapshot,
    TelemetryWindow,
)


_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "fp8": 1,
    "int4": 0.5,
}


class AnalyticalPerformanceModel:
    """Produces transparent upper-bound estimates from collected facts."""

    def estimate(self, context: SystemContextSnapshot) -> PerformanceEnvelope:
        model = context.model
        hardware = context.hardware
        assumptions: list[str] = []
        weight_bytes = model.weight_bytes
        dtype_bytes = _DTYPE_BYTES.get((model.dtype or "").lower())
        if weight_bytes is None and model.parameter_count and dtype_bytes:
            weight_bytes = int(model.parameter_count * dtype_bytes)
            assumptions.append("weight_bytes estimated from parameter_count and dtype")

        total_flops = sum(
            device.compute_flops_by_dtype.get(model.dtype or "", 0.0) * device.count
            for device in hardware.accelerators
        )
        compute_bound = None
        if total_flops > 0 and model.parameter_count:
            compute_bound = total_flops / (2.0 * model.parameter_count)
            assumptions.append("decode compute uses approximately 2 FLOPs per parameter per token")

        total_bandwidth = sum(
            (device.hbm_bandwidth_bytes_per_second or 0.0) * device.count
            for device in hardware.accelerators
        )
        memory_bound = total_bandwidth / weight_bytes if total_bandwidth > 0 and weight_bytes else None
        if memory_bound is not None:
            assumptions.append("decode memory bound assumes one aggregate weight read per token")

        kv_capacity = self._kv_capacity(context, weight_bytes, dtype_bytes)
        return PerformanceEnvelope(
            weight_bytes=weight_bytes,
            estimated_kv_capacity_tokens=kv_capacity,
            compute_upper_bound_tokens_per_second=compute_bound,
            memory_upper_bound_tokens_per_second=memory_bound,
            assumptions=tuple(assumptions),
        )

    @staticmethod
    def _kv_capacity(
        context: SystemContextSnapshot,
        weight_bytes: int | None,
        dtype_bytes: float | None,
    ) -> int | None:
        model = context.model
        total_hbm = sum(
            (device.hbm_bytes_per_device or 0) * device.count
            for device in context.hardware.accelerators
        )
        if not all((total_hbm, weight_bytes, dtype_bytes, model.layer_count, model.hidden_size)):
            return None
        available = max(0, total_hbm - weight_bytes)
        head_ratio = 1.0
        if model.kv_heads and model.attention_heads:
            head_ratio = model.kv_heads / model.attention_heads
        bytes_per_token = 2 * model.layer_count * model.hidden_size * head_ratio * dtype_bytes
        return int(available / bytes_per_token) if bytes_per_token > 0 else None


class RuleBasedBottleneckDiagnoser:
    """Evidence-first baseline diagnoser; rules can later be augmented by experience."""

    def diagnose(
        self,
        objective: PerformanceObjective,
        context: SystemContextSnapshot,
        metrics: dict[str, float],
        telemetry: TelemetryWindow | None,
        envelope: PerformanceEnvelope,
    ) -> tuple[BottleneckHypothesis, ...]:
        evidence = telemetry.metrics if telemetry else {}
        hypotheses: list[BottleneckHypothesis] = []
        if evidence.get("hbm_capacity_utilization", 0.0) >= 0.9:
            hypotheses.append(self._hypothesis("kv_capacity", 0.9, telemetry))
        if evidence.get("hbm_bandwidth_utilization", 0.0) >= 0.8:
            hypotheses.append(self._hypothesis("hbm_bandwidth", 0.85, telemetry))
        if evidence.get("accelerator_compute_utilization", 0.0) >= 0.85:
            hypotheses.append(self._hypothesis("compute", 0.85, telemetry))
        if evidence.get("interconnect_utilization", 0.0) >= 0.8:
            hypotheses.append(self._hypothesis("interconnect", 0.8, telemetry))
        if evidence.get("host_cpu_utilization", 0.0) >= 0.9:
            hypotheses.append(self._hypothesis("host", 0.8, telemetry))
        if evidence.get("benchmark_dispatch_lag_p95_ms", 0.0) > 10.0:
            hypotheses.append(self._hypothesis("load_generator", 0.8, telemetry))
        if hypotheses:
            return tuple(sorted(hypotheses, key=lambda item: item.confidence, reverse=True))
        return (
            BottleneckHypothesis(
                category="insufficient_evidence",
                summary="baseline metrics require correlated telemetry",
                confidence=0.2,
                evidence_refs=(),
                missing_evidence=(
                    MissingEvidence(
                        field="telemetry",
                        reason="no discriminating utilization signal is available",
                        requested_capability="telemetry.collect",
                    ),
                ),
            ),
        )

    @staticmethod
    def _hypothesis(
        category: str, confidence: float, telemetry: TelemetryWindow | None
    ) -> BottleneckHypothesis:
        return BottleneckHypothesis(
            category=category,
            summary=f"telemetry indicates a {category} bottleneck",
            confidence=confidence,
            evidence_refs=(f"telemetry:{telemetry.run_id}",) if telemetry else (),
        )


class ConservativeCandidatePlanner:
    """Ranks applicable engine features and emits one reversible candidate."""

    def plan(
        self,
        objective: PerformanceObjective,
        context: SystemContextSnapshot,
        hypotheses: tuple[BottleneckHypothesis, ...],
        features: tuple[EngineFeature, ...],
    ) -> tuple[OptimizationCandidate, ...]:
        if not hypotheses:
            return ()
        category = hypotheses[0].category
        applicable = [
            feature
            for feature in features
            if category in feature.expected_effects
            and self._matches_context(feature, context)
        ]
        applicable.sort(key=lambda item: (self._risk_rank(item.risk), item.restart_required))
        if not applicable:
            return ()
        feature = applicable[0]
        current = context.engine.effective_parameters.get(feature.parameter)
        proposed = feature.prerequisites.get("proposed_value")
        return (
            OptimizationCandidate(
                candidate_id=f"candidate:{feature.feature_id}",
                feature_id=feature.feature_id,
                parameter=feature.parameter,
                previous_value=current,
                proposed_value=proposed,
                rationale=feature.expected_effects[category],
                target_metrics=feature.validation_metrics,
                evidence_refs=hypotheses[0].evidence_refs,
                risk=feature.risk,
                restart_required=feature.restart_required,
                rollback_value=current,
            ),
        )

    @staticmethod
    def _matches_context(feature: EngineFeature, context: SystemContextSnapshot) -> bool:
        if feature.engine.casefold() != context.engine.name.casefold():
            return False
        accelerator = feature.prerequisites.get("accelerator_model")
        if accelerator and all(
            device.model != accelerator for device in context.hardware.accelerators
        ):
            return False
        return True

    @staticmethod
    def _risk_rank(risk: RiskLevel) -> int:
        return {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}[risk]


class DeterministicComparisonBuilder:
    def build(
        self,
        experiment_id: str,
        baseline_run_id: str,
        candidate_run_id: str,
        objective: PerformanceObjective,
        baseline: dict[str, float],
        candidate: dict[str, float],
        evidence_refs: tuple[str, ...] = (),
    ) -> ComparisonSummary:
        keys = baseline.keys() & candidate.keys()
        absolute = {key: candidate[key] - baseline[key] for key in keys}
        relative = {
            key: (absolute[key] / baseline[key] * 100.0) if baseline[key] else None
            for key in keys
        }
        constraint_results = {
            self._constraint_key(item): self._compare(
                candidate.get(self._metric_key(item.metric, item.aggregation.value)),
                item.operator,
                item.value,
            )
            for item in objective.constraints
        }
        conclusion = "constraints_satisfied" if all(constraint_results.values()) else "constraints_violated"
        return ComparisonSummary(
            experiment_id=experiment_id,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            baseline_metrics=dict(baseline),
            candidate_metrics=dict(candidate),
            absolute_delta=absolute,
            relative_delta_percent=relative,
            constraint_results=constraint_results,
            conclusion=conclusion,
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def _metric_key(metric: str, aggregation: str) -> str:
        return f"{metric}_{aggregation}"

    @staticmethod
    def _constraint_key(item: Any) -> str:
        return f"{item.metric}_{item.aggregation.value}{item.operator}{item.value}{item.unit}"

    @staticmethod
    def _compare(actual: float | None, operator: str, target: float) -> bool:
        if actual is None:
            return False
        return {
            ">": actual > target,
            ">=": actual >= target,
            "<": actual < target,
            "<=": actual <= target,
            "==": actual == target,
        }[operator]

