from coco_agent.domain import (
    Aggregation,
    AcceleratorDevice,
    ContextValidator,
    DeploymentContext,
    EngineContext,
    ExperimentBudget,
    HardwareContext,
    MetricConstraint,
    ModelContext,
    ObjectiveValidator,
    OptimizationDirection,
    PerformanceObjective,
    SystemContextSnapshot,
    TargetIntent,
    WorkloadIntent,
)


def objective(**overrides):
    values = {
        "task_id": "tuning-1",
        "primary_metric": "throughput_requests_per_second",
        "direction": OptimizationDirection.MAXIMIZE,
        "workload": WorkloadIntent("workloads/production.jsonl", "sha256:abc"),
        "target": TargetIntent("local/Qwen", "deployment:primary"),
        "constraints": (
            MetricConstraint("ttft", "<=", 800.0, "ms", Aggregation.P95),
            MetricConstraint("tpot", "<=", 40.0, "ms/token", Aggregation.P95),
        ),
        "budget": ExperimentBudget(max_runs=3),
    }
    values.update(overrides)
    return PerformanceObjective(**values)


def context(**overrides):
    values = {
        "snapshot_id": "ctx-1",
        "captured_at": "2026-08-29T12:00:00Z",
        "model": ModelContext(model_ref="local/Qwen", dtype="bfloat16"),
        "engine": EngineContext(name="engine", version="1.0"),
        "deployment": DeploymentContext(kind="process", endpoint_ref="target:primary"),
        "hardware": HardwareContext(
            accelerators=(AcceleratorDevice("vendor", "accelerator", 8),)
        ),
    }
    values.update(overrides)
    return SystemContextSnapshot(**values)


def test_objective_expresses_percentile_sla():
    value = objective()
    assert value.direction is OptimizationDirection.MAXIMIZE
    assert value.constraints[0].aggregation is Aggregation.P95
    assert ObjectiveValidator().validate(value) == ()


def test_objective_validator_reports_structured_missing_fields():
    value = objective(
        task_id="",
        workload=WorkloadIntent(""),
        target=TargetIntent("local/Qwen", "deployment:primary"),
        budget=ExperimentBudget(max_runs=0),
    )
    issues = ObjectiveValidator().validate(value)
    assert {issue.field for issue in issues} == {
        "task_id",
        "workload.workload_ref",
        "budget.max_runs",
    }


def test_context_requires_engine_version_and_accelerator():
    value = context(
        engine=EngineContext(name="engine", version=""),
        hardware=HardwareContext(accelerators=()),
    )
    issues = ContextValidator().validate(value)
    assert {issue.field for issue in issues} == {"engine", "hardware.accelerators"}


def test_context_contract_keeps_engine_and_hardware_separate():
    value = context()
    assert value.engine.name == "engine"
    assert value.hardware.accelerators[0].count == 8
