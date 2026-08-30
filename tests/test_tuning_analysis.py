from coco_agent.domain import (
    Aggregation,
    AcceleratorDevice,
    DeploymentContext,
    EngineContext,
    EngineFeature,
    ExperimentBudget,
    HardwareContext,
    MetricConstraint,
    ModelContext,
    OptimizationDirection,
    PerformanceObjective,
    RiskLevel,
    SystemContextSnapshot,
    TargetIntent,
    WorkloadIntent,
)
from coco_agent.domain.analysis import (
    AnalyticalPerformanceModel,
    ConservativeCandidatePlanner,
    DeterministicComparisonBuilder,
    RuleBasedBottleneckDiagnoser,
)
from coco_agent.domain.tuning import TelemetryWindow


def fixture_context():
    return SystemContextSnapshot(
        snapshot_id="ctx-1",
        captured_at="2026-08-29T12:00:00Z",
        model=ModelContext(
            model_ref="model",
            parameter_count=7_000_000_000,
            layer_count=32,
            hidden_size=4096,
            attention_heads=32,
            kv_heads=8,
            dtype="bfloat16",
        ),
        engine=EngineContext(
            name="engine",
            version="1.0",
            effective_parameters={"parallel_size": 1},
        ),
        deployment=DeploymentContext(kind="process", endpoint_ref="target"),
        hardware=HardwareContext(
            accelerators=(
                AcceleratorDevice(
                    vendor="vendor",
                    model="device",
                    count=2,
                    hbm_bytes_per_device=32_000_000_000,
                    hbm_bandwidth_bytes_per_second=1_000_000_000_000,
                    compute_flops_by_dtype={"bfloat16": 100_000_000_000_000},
                ),
            )
        ),
    )


def fixture_objective():
    return PerformanceObjective(
        task_id="task",
        primary_metric="throughput",
        direction=OptimizationDirection.MAXIMIZE,
        workload=WorkloadIntent("workload", "hash"),
        target=TargetIntent("model", "target"),
        constraints=(MetricConstraint("ttft", "<=", 800, "ms", Aggregation.P95),),
        budget=ExperimentBudget(max_runs=2),
    )


def test_performance_model_produces_reproducible_capacity_estimates():
    envelope = AnalyticalPerformanceModel().estimate(fixture_context())
    assert envelope.weight_bytes == 14_000_000_000
    assert envelope.estimated_kv_capacity_tokens is not None
    assert envelope.compute_upper_bound_tokens_per_second is not None
    assert envelope.memory_upper_bound_tokens_per_second is not None


def test_diagnoser_uses_correlated_telemetry():
    context = fixture_context()
    envelope = AnalyticalPerformanceModel().estimate(context)
    telemetry = TelemetryWindow(
        run_id="run-1",
        started_at="start",
        finished_at="finish",
        metrics={"hbm_bandwidth_utilization": 0.91},
    )
    result = RuleBasedBottleneckDiagnoser().diagnose(
        fixture_objective(), context, {}, telemetry, envelope
    )
    assert result[0].category == "hbm_bandwidth"
    assert result[0].evidence_refs == ("telemetry:run-1",)


def test_candidate_planner_selects_low_risk_applicable_feature():
    context = fixture_context()
    hypothesis = RuleBasedBottleneckDiagnoser().diagnose(
        fixture_objective(),
        context,
        {},
        TelemetryWindow("run-1", "start", "finish", {"hbm_bandwidth_utilization": 0.9}),
        AnalyticalPerformanceModel().estimate(context),
    )
    feature = EngineFeature(
        feature_id="parallel-2",
        engine="engine",
        version_spec=">=1.0",
        parameter="parallel_size",
        description="increase parallel size",
        prerequisites={"accelerator_model": "device", "proposed_value": 2},
        conflicts=(),
        expected_effects={"hbm_bandwidth": "increase aggregate memory bandwidth"},
        risk=RiskLevel.LOW,
        restart_required=True,
        validation_metrics=("throughput", "ttft_p95"),
        source_url="https://docs.example/feature",
        retrieved_at="2026-08-29",
    )
    candidates = ConservativeCandidatePlanner().plan(
        fixture_objective(), context, hypothesis, (feature,)
    )
    assert candidates[0].previous_value == 1
    assert candidates[0].proposed_value == 2
    assert candidates[0].rollback_value == 1


def test_comparison_builder_evaluates_percentile_constraints():
    report = DeterministicComparisonBuilder().build(
        "experiment-1",
        "baseline",
        "candidate",
        fixture_objective(),
        {"throughput": 80.0, "ttft_p95": 900.0},
        {"throughput": 100.0, "ttft_p95": 700.0},
    )
    assert report.relative_delta_percent["throughput"] == 25.0
    assert report.conclusion == "constraints_satisfied"
