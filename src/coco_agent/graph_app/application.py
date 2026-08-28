"""LangGraph-backed tuning workflow application."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..domain.contracts import (
    ActionRequest,
    ActionResult,
    AuthorizationVerdict,
    Criterion,
    GoalSpec,
    MetricAnalysis,
    ParameterRecommendation,
    TaskStatus,
    VerificationVerdict,
)
from .state import TuningGraphState


class TuningApplication:
    def __init__(
        self,
        *,
        checkpointer: Any,
        capability: Any,
        authorization: Any,
        verifier: Any,
        experience: Any,
        tuning_policy: Any,
        report_builder: Any,
    ) -> None:
        self._capability = capability
        self._authorization = authorization
        self._verifier = verifier
        self._experience = experience
        self._tuning_policy = tuning_policy
        self._report_builder = report_builder
        self.graph = self._build_graph().compile(checkpointer=checkpointer)

    @staticmethod
    def _goal(raw: dict[str, Any]) -> GoalSpec:
        return GoalSpec(
            task_id=raw["task_id"],
            objective=raw["objective"],
            acceptance_criteria=tuple(Criterion(**item) for item in raw["acceptance_criteria"]),
            inputs=dict(raw.get("inputs", {})),
            constraints=dict(raw.get("constraints", {})),
        )

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(TuningGraphState)
        builder.add_node("normalize_goal", self._normalize_goal)
        builder.add_node("observe_environment", self._observe_environment)
        builder.add_node("prepare_baseline_benchmark", self._prepare_baseline_benchmark)
        builder.add_node("execute_baseline_benchmark", self._execute_baseline_benchmark)
        builder.add_node("analyze_metrics", self._analyze_metrics)
        builder.add_node("retrieve_experience", self._retrieve_experience)
        builder.add_node("recommend_parameter", self._recommend_parameter)
        builder.add_node("prepare_candidate_benchmark", self._prepare_candidate_benchmark)
        builder.add_node("authorize_action", self._authorize_action)
        builder.add_node("request_approval", self._request_approval)
        builder.add_node("execute_candidate_benchmark", self._execute_candidate_benchmark)
        builder.add_node("verify_action", self._verify_action)
        builder.add_node("build_comparison_report", self._build_comparison_report)
        builder.add_node("record_experiments", self._record_experiments)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "normalize_goal")
        builder.add_conditional_edges(
            "normalize_goal",
            lambda state: "continue" if state["status"] == TaskStatus.RUNNING.value else "finalize",
            {"continue": "observe_environment", "finalize": "finalize"},
        )
        builder.add_edge("observe_environment", "prepare_baseline_benchmark")
        builder.add_edge("prepare_baseline_benchmark", "execute_baseline_benchmark")
        builder.add_edge("execute_baseline_benchmark", "analyze_metrics")
        builder.add_edge("analyze_metrics", "retrieve_experience")
        builder.add_edge("retrieve_experience", "recommend_parameter")
        builder.add_edge("recommend_parameter", "prepare_candidate_benchmark")
        builder.add_edge("prepare_candidate_benchmark", "authorize_action")
        builder.add_conditional_edges(
            "authorize_action",
            lambda state: state["status"],
            {
                TaskStatus.RUNNING.value: "execute_candidate_benchmark",
                TaskStatus.WAITING_APPROVAL.value: "request_approval",
                TaskStatus.FAILED.value: "finalize",
            },
        )
        builder.add_conditional_edges(
            "request_approval",
            lambda state: "execute" if state["status"] == TaskStatus.RUNNING.value else "finalize",
            {"execute": "execute_candidate_benchmark", "finalize": "finalize"},
        )
        builder.add_edge("execute_candidate_benchmark", "verify_action")
        builder.add_edge("verify_action", "build_comparison_report")
        builder.add_edge("build_comparison_report", "record_experiments")
        builder.add_edge("record_experiments", "finalize")
        builder.add_edge("finalize", END)
        return builder

    def _normalize_goal(self, state: TuningGraphState) -> dict[str, Any]:
        goal = self._goal(state["goal"])
        if not goal.acceptance_criteria:
            return {
                "status": TaskStatus.FAILED.value,
                "terminal_result": {"reason": "acceptance criteria are required"},
            }
        return {"task_id": goal.task_id, "status": TaskStatus.RUNNING.value}

    def _observe_environment(self, state: TuningGraphState) -> dict[str, Any]:
        return {"environment": self._capability.inspect_environment(self._goal(state["goal"]))}

    def _prepare_baseline_benchmark(self, state: TuningGraphState) -> dict[str, Any]:
        goal = self._goal(state["goal"])
        action = ActionRequest(
            capability="run_benchmark",
            arguments={
                "phase": "baseline",
                "config": dict(goal.inputs.get("baseline_config", {"max_num_seqs": 128})),
                "mock_metrics": dict(
                    goal.inputs.get("baseline_metrics", {"throughput": 80.0})
                ),
            },
            operation_id=f"op-{goal.task_id}-baseline",
            idempotency_key=f"{goal.task_id}:benchmark:baseline",
            risk="low",
        )
        return {"baseline_action": action.to_dict()}

    def _execute_baseline_benchmark(self, state: TuningGraphState) -> dict[str, Any]:
        request = ActionRequest(**state["baseline_action"])
        existing = self._capability.status(request.operation_id)
        result = existing or self._capability.execute(request)
        return {"baseline_result": result.to_dict()}

    @staticmethod
    def _metrics(result: dict[str, Any]) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in result.get("output", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    def _analyze_metrics(self, state: TuningGraphState) -> dict[str, Any]:
        analysis = self._tuning_policy.analyze(
            self._goal(state["goal"]),
            self._metrics(state["baseline_result"]),
        )
        return {"metric_analysis": analysis.to_dict()}

    def _retrieve_experience(self, state: TuningGraphState) -> dict[str, Any]:
        matches = self._experience.retrieve({
            "goal": state["goal"],
            "environment": state["environment"],
        })
        return {"experience_matches": matches}

    def _recommend_parameter(self, state: TuningGraphState) -> dict[str, Any]:
        goal = self._goal(state["goal"])
        analysis = MetricAnalysis(**state["metric_analysis"])
        recommendation = self._tuning_policy.recommend(goal, analysis)
        return {"recommendation": recommendation.to_dict()}

    def _prepare_candidate_benchmark(self, state: TuningGraphState) -> dict[str, Any]:
        goal = self._goal(state["goal"])
        recommendation = ParameterRecommendation(**state["recommendation"])
        candidate_config = dict(goal.inputs.get("baseline_config", {"max_num_seqs": 128}))
        candidate_config[recommendation.parameter] = recommendation.recommended_value
        risk = str(goal.constraints.get("risk", "low"))
        action = ActionRequest(
            capability="run_benchmark",
            arguments={
                "phase": "candidate",
                "config": candidate_config,
                "mock_metrics": dict(
                    goal.inputs.get(
                        "candidate_metrics",
                        goal.inputs.get("mock_metrics", {"throughput": 120.0}),
                    )
                ),
            },
            operation_id=f"op-{goal.task_id}-candidate",
            idempotency_key=f"{goal.task_id}:benchmark:candidate",
            risk=risk,
        )
        return {"candidate_action": action.to_dict(), "pending_action": action.to_dict()}

    def _authorize_action(self, state: TuningGraphState) -> dict[str, Any]:
        request = ActionRequest(**state["pending_action"])
        decision = self._authorization.authorize(request)
        authorization = {
            "verdict": decision.verdict.value,
            "reason": decision.reason,
            "policy_version": decision.policy_version,
        }
        if decision.verdict == AuthorizationVerdict.DENY:
            return {"authorization": authorization, "status": TaskStatus.FAILED.value}
        if decision.verdict == AuthorizationVerdict.REQUIRE_APPROVAL:
            return {
                "authorization": authorization,
                "status": TaskStatus.WAITING_APPROVAL.value,
            }
        return {"authorization": authorization, "status": TaskStatus.RUNNING.value}

    def _request_approval(self, state: TuningGraphState) -> dict[str, Any]:
        answer = interrupt({
            "type": "action_approval",
            "task_id": state["task_id"],
            "action": state["pending_action"],
            "reason": state["authorization"]["reason"],
        })
        approved = bool(answer.get("approved")) if isinstance(answer, dict) else bool(answer)
        authorization = dict(state["authorization"])
        authorization["approved"] = approved
        return {
            "authorization": authorization,
            "status": TaskStatus.RUNNING.value if approved else TaskStatus.FAILED.value,
        }

    def _execute_candidate_benchmark(self, state: TuningGraphState) -> dict[str, Any]:
        request = ActionRequest(**state["pending_action"])
        existing = self._capability.status(request.operation_id)
        result = existing or self._capability.execute(request)
        return {"candidate_result": result.to_dict(), "action_result": result.to_dict()}

    def _verify_action(self, state: TuningGraphState) -> dict[str, Any]:
        result = ActionResult(**state["candidate_result"])
        verification = self._verifier.verify(self._goal(state["goal"]), result)
        return {"verification": verification.to_dict()}

    def _build_comparison_report(self, state: TuningGraphState) -> dict[str, Any]:
        from ..domain.contracts import Verification

        verification_raw = state["verification"]
        verification = Verification(
            verdict=VerificationVerdict(verification_raw["verdict"]),
            evidence=dict(verification_raw["evidence"]),
            reason=verification_raw["reason"],
        )
        report = self._report_builder.build(
            self._metrics(state["baseline_result"]),
            self._metrics(state["candidate_result"]),
            ParameterRecommendation(**state["recommendation"]),
            verification,
        )
        return {"comparison_report": report.to_dict()}

    def _record_experiments(self, state: TuningGraphState) -> dict[str, Any]:
        baseline_id = self._experience.record_experiment({
            "task_id": state["task_id"],
            "environment": state["environment"],
            "phase": "baseline",
            "action": state["baseline_action"],
            "result": state["baseline_result"],
        })
        candidate_id = self._experience.record_experiment({
            "task_id": state["task_id"],
            "environment": state["environment"],
            "phase": "candidate",
            "action": state["candidate_action"],
            "result": state["candidate_result"],
            "verification": state["verification"],
            "comparison_report": state["comparison_report"],
        })
        return {
            "experiment_id": candidate_id,
            "experiment_ids": [baseline_id, candidate_id],
        }

    def _finalize(self, state: TuningGraphState) -> dict[str, Any]:
        if state.get("terminal_result"):
            return {}
        verification = state.get("verification", {})
        succeeded = verification.get("verdict") == VerificationVerdict.PASSED.value
        status = TaskStatus.SUCCEEDED if succeeded else TaskStatus.FAILED
        return {
            "status": status.value,
            "terminal_result": {
                "success": succeeded,
                "verification": verification,
                "experiment_id": state.get("experiment_id"),
                "experiment_ids": state.get("experiment_ids", []),
                "comparison_report": state.get("comparison_report"),
            },
        }

    @staticmethod
    def config(task_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": task_id}}

    def start(self, goal: GoalSpec) -> dict[str, Any]:
        return self.graph.invoke(
            {"task_id": goal.task_id, "goal": goal.to_dict(), "status": TaskStatus.CREATED.value},
            config=self.config(goal.task_id),
        )

    def resume(self, task_id: str, *, approved: bool) -> dict[str, Any]:
        return self.graph.invoke(
            Command(resume={"approved": approved}),
            config=self.config(task_id),
        )

    def inspect(self, task_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self.config(task_id))
        return dict(snapshot.values)
