"""Non-production components for the public runnable example and contract tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from .domain.contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    ComparisonReport,
    MetricAnalysis,
    ParameterRecommendation,
    Verification,
)


class AllowLowRiskActions:
    def authorize(self, request):
        verdict = (
            AuthorizationVerdict.REQUIRE_APPROVAL
            if request.risk == "high"
            else AuthorizationVerdict.ALLOW
        )
        return AuthorizationDecision(verdict, "demo risk policy")


class NumericGoalVerifier:
    _operators = {
        ">=": lambda actual, target: actual >= target,
        "<=": lambda actual, target: actual <= target,
        ">": lambda actual, target: actual > target,
        "<": lambda actual, target: actual < target,
        "==": lambda actual, target: actual == target,
    }

    def verify(self, goal, result):
        from .domain.contracts import VerificationVerdict

        checks = []
        for criterion in goal.acceptance_criteria:
            actual = result.output.get(criterion.metric)
            operator = self._operators.get(criterion.operator)
            passed = actual is not None and operator is not None and operator(actual, criterion.target)
            checks.append({"metric": criterion.metric, "passed": bool(passed)})
        passed = result.success and bool(checks) and all(item["passed"] for item in checks)
        return Verification(
            VerificationVerdict.PASSED if passed else VerificationVerdict.FAILED,
            {"checks": checks},
            "demo acceptance check",
        )


class DoubleConcurrencyPolicy:
    def analyze(self, goal, baseline_metrics):
        throughput = float(baseline_metrics.get("throughput", 0.0))
        target = next((x.target for x in goal.acceptance_criteria if x.metric == "throughput"), throughput)
        below = throughput < target
        return MetricAnalysis(
            "throughput_below_target" if below else "target_already_met",
            {"throughput": throughput, "target_throughput": target},
            "baseline throughput is below target" if below else "baseline throughput already meets target",
        )

    def recommend(self, goal, analysis):
        previous = int(goal.inputs.get("baseline_config", {}).get("max_num_seqs", 128))
        value = previous * 2 if analysis.bottleneck == "throughput_below_target" else previous
        return ParameterRecommendation("max_num_seqs", previous, value, analysis.summary)


class SimpleReportBuilder:
    def build(self, baseline, candidate, recommendation, verification):
        common = sorted(set(baseline).intersection(candidate))
        absolute = {key: float(candidate[key]) - float(baseline[key]) for key in common}
        relative = {
            key: absolute[key] / float(baseline[key]) * 100.0 if float(baseline[key]) else None
            for key in common
        }
        return ComparisonReport(
            {key: float(value) for key, value in baseline.items()},
            {key: float(value) for key, value in candidate.items()},
            absolute,
            relative,
            recommendation.to_dict(),
            verification.verdict.value,
        )


class InMemoryExperience:
    def __init__(self):
        self.experiments: dict[str, dict[str, Any]] = {}

    def retrieve(self, context):
        return []

    def record_experiment(self, record):
        experiment_id = f"exp-{uuid4().hex}"
        self.experiments[experiment_id] = deepcopy(record) | {"lifecycle": "raw"}
        return experiment_id

    def get_experiment(self, experiment_id):
        item = self.experiments.get(experiment_id)
        return deepcopy(item) if item else None
