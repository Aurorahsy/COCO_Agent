from coco_agent.context import CompositeContextCollector
from coco_agent.domain import (
    AcceleratorDevice,
    DeploymentContext,
    EngineContext,
    ExperimentBudget,
    HardwareContext,
    ModelContext,
    OptimizationDirection,
    PerformanceObjective,
    TargetIntent,
    WorkloadIntent,
)
from coco_agent.domain.tuning import ArtifactRef, BenchmarkRunLink
from coco_agent.experiments import InMemoryExperimentManifestRepository, ManifestService


class Inspector:
    def __init__(self, value):
        self.value = value
        self.refs = []

    def inspect(self, ref):
        self.refs.append(ref)
        return self.value


def objective():
    return PerformanceObjective(
        task_id="task",
        primary_metric="throughput",
        direction=OptimizationDirection.MAXIMIZE,
        workload=WorkloadIntent("workload", "workload-hash"),
        target=TargetIntent("model-ref", "deployment-ref"),
        budget=ExperimentBudget(max_runs=2),
    )


def collector():
    return CompositeContextCollector(
        model=Inspector(ModelContext("model-ref", dtype="bfloat16")),
        engine=Inspector(EngineContext("engine", "1.0")),
        deployment=Inspector(DeploymentContext("process", "endpoint")),
        hardware=Inspector(
            HardwareContext((AcceleratorDevice("vendor", "device", 2),))
        ),
        id_factory=lambda: "context-1",
        clock=lambda: "2026-08-29T12:00:00Z",
    )


def test_context_collector_builds_stable_fingerprint():
    first = collector().collect(objective())
    second = collector().collect(objective())
    assert first.fingerprint == second.fingerprint
    assert first.model.model_ref == "model-ref"
    assert first.engine.version == "1.0"


def test_manifest_service_associates_run_and_agent_evidence():
    repository = InMemoryExperimentManifestRepository()
    service = ManifestService(
        repository,
        id_factory=lambda: "experiment-1",
        clock=lambda: "2026-08-29T12:01:00Z",
    )
    manifest = service.create(objective(), collector().collect(objective()))
    manifest = service.attach_benchmark_run(
        manifest.experiment_id,
        BenchmarkRunLink(
            phase="baseline",
            run_id="run-1",
            receipt_ref="artifact:receipt",
            workload_hash="workload-hash",
            started_at="start",
            finished_at="finish",
        ),
    )
    manifest = service.attach_telemetry(
        manifest.experiment_id,
        ArtifactRef("telemetry-1", "telemetry", "artifact:telemetry", "sha256"),
    )
    assert manifest.benchmark_runs[0].run_id == "run-1"
    assert manifest.telemetry_artifacts[0].artifact_id == "telemetry-1"
    assert repository.get("experiment-1") == manifest

