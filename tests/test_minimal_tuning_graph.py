from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from deployopt_agent.capabilities import MockTuningCapability, SqliteOperationStore
from deployopt_agent.domain.contracts import Criterion, GoalSpec, TaskStatus
from deployopt_agent.demo import (
    AllowLowRiskActions,
    DoubleConcurrencyPolicy,
    InMemoryExperience,
    NumericGoalVerifier,
    SimpleReportBuilder,
)
from deployopt_agent.graph_app import TuningApplication


def build_app(tmp_path, *, experience=None):
    stack = ExitStack()
    checkpointer = stack.enter_context(
        SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite"))
    )
    capability = MockTuningCapability(SqliteOperationStore(tmp_path / "operations.sqlite"))
    app = TuningApplication(
        checkpointer=checkpointer,
        capability=capability,
        authorization=AllowLowRiskActions(),
        verifier=NumericGoalVerifier(),
        experience=experience or InMemoryExperience(),
        tuning_policy=DoubleConcurrencyPolicy(),
        report_builder=SimpleReportBuilder(),
    )
    return stack, app, capability


def goal(task_id="task-1", *, risk="low", throughput=120.0):
    return GoalSpec(
        task_id=task_id,
        objective="meet throughput target",
        acceptance_criteria=(Criterion("throughput", ">=", 100.0),),
        inputs={
            "hardware": "mock-gpu",
            "model": "mock-model",
            "baseline_config": {"max_num_seqs": 128},
            "baseline_metrics": {"throughput": 80.0, "latency_p99_ms": 220.0},
            "candidate_metrics": {
                "throughput": throughput,
                "latency_p99_ms": 170.0,
            },
        },
        constraints={"risk": risk},
    )


def test_rule_driven_graph_completes_without_llm(tmp_path):
    stack, app, capability = build_app(tmp_path)
    with stack:
        result = app.start(goal())
        assert result["status"] == TaskStatus.SUCCEEDED.value
        assert result["terminal_result"]["success"] is True
        assert result["experiment_id"].startswith("exp-")
        assert capability.execution_count == 2


def test_high_risk_action_interrupts_and_resumes(tmp_path):
    stack, app, capability = build_app(tmp_path)
    with stack:
        interrupted = app.start(goal("task-approval", risk="high"))
        assert "__interrupt__" in interrupted
        assert capability.execution_count == 1
        assert app.inspect("task-approval")["status"] == TaskStatus.WAITING_APPROVAL.value

        result = app.resume("task-approval", approved=True)
        assert result["status"] == TaskStatus.SUCCEEDED.value
        assert capability.execution_count == 2


def test_denied_approval_finishes_without_side_effect(tmp_path):
    stack, app, capability = build_app(tmp_path)
    with stack:
        app.start(goal("task-denied", risk="high"))
        result = app.resume("task-denied", approved=False)
        assert result["status"] == TaskStatus.FAILED.value
        assert capability.execution_count == 1


def test_checkpoint_survives_new_application_instance(tmp_path):
    stack1, app1, capability1 = build_app(tmp_path)
    with stack1:
        interrupted = app1.start(goal("task-restart", risk="high"))
        assert "__interrupt__" in interrupted
        assert capability1.execution_count == 1

    stack2, app2, capability2 = build_app(tmp_path)
    with stack2:
        result = app2.resume("task-restart", approved=True)
        assert result["status"] == TaskStatus.SUCCEEDED.value
        assert capability2.execution_count == 1


def test_operation_store_prevents_duplicate_side_effect(tmp_path):
    stack1, app1, capability1 = build_app(tmp_path)
    with stack1:
        first = app1.start(goal("task-idempotent"))
        assert first["status"] == TaskStatus.SUCCEEDED.value
        assert capability1.execution_count == 2

    stack2, app2, capability2 = build_app(tmp_path)
    with stack2:
        request = first["pending_action"]
        from deployopt_agent.domain.contracts import ActionRequest

        result = capability2.execute(ActionRequest(**request))
        assert result.reused is True
        assert capability2.execution_count == 0


def test_failed_criterion_is_recorded_as_raw_experiment(tmp_path):
    experience = InMemoryExperience()
    stack, app, _ = build_app(tmp_path, experience=experience)
    with stack:
        result = app.start(goal("task-fail", throughput=80.0))
        assert result["status"] == TaskStatus.FAILED.value
        experiment = experience.get_experiment(result["experiment_id"])
        assert experiment["lifecycle"] == "raw"
        assert experiment["verification"]["verdict"] == "failed"


def test_goal_without_acceptance_criteria_stops_before_observation(tmp_path):
    stack, app, capability = build_app(tmp_path)
    invalid_goal = GoalSpec(
        task_id="task-no-criteria",
        objective="underspecified goal",
        acceptance_criteria=(),
    )
    with stack:
        result = app.start(invalid_goal)
        assert result["status"] == TaskStatus.FAILED.value
        assert result["terminal_result"]["reason"] == "acceptance criteria are required"
        assert "environment" not in result
        assert capability.execution_count == 0


def test_minimal_tuning_chain_outputs_parameter_and_comparison(tmp_path):
    stack, app, capability = build_app(tmp_path)
    with stack:
        result = app.start(goal("task-minimal-chain"))

        assert result["environment"]["hardware"] == "mock-gpu"
        assert result["baseline_result"]["output"]["throughput"] == 80.0
        assert result["metric_analysis"]["bottleneck"] == "throughput_below_target"
        assert result["recommendation"] == {
            "parameter": "max_num_seqs",
            "previous_value": 128,
            "recommended_value": 256,
            "rationale": "baseline throughput is below target",
        }
        assert result["candidate_result"]["output"]["throughput"] == 120.0
        report = result["terminal_result"]["comparison_report"]
        assert report["baseline_metrics"]["throughput"] == 80.0
        assert report["candidate_metrics"]["throughput"] == 120.0
        assert report["absolute_delta"]["throughput"] == 40.0
        assert report["relative_delta_pct"]["throughput"] == 50.0
        assert report["verdict"] == "passed"
        assert len(result["experiment_ids"]) == 2
        assert capability.execution_count == 2


def test_operation_store_releases_sqlite_file(tmp_path):
    database = tmp_path / "releasable.sqlite"
    store = SqliteOperationStore(database)
    assert store.get_by_operation("missing") is None
    Path(database).unlink()
    assert not database.exists()
