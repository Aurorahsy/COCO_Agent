"""Experiment manifest lifecycle owned by COCO_Agent."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from ..domain.tuning import (
    ArtifactRef,
    BenchmarkRunLink,
    ExperimentManifest,
    PerformanceObjective,
    SystemContextSnapshot,
)


class InMemoryExperimentManifestRepository:
    def __init__(self) -> None:
        self._items: dict[str, ExperimentManifest] = {}

    def save(self, manifest: ExperimentManifest) -> None:
        self._items[manifest.experiment_id] = manifest

    def get(self, experiment_id: str) -> ExperimentManifest | None:
        return self._items.get(experiment_id)


class ManifestService:
    def __init__(self, repository, *, id_factory: Callable[[], str], clock: Callable[[], str]):
        self._repository = repository
        self._id_factory = id_factory
        self._clock = clock

    def create(
        self, objective: PerformanceObjective, context: SystemContextSnapshot
    ) -> ExperimentManifest:
        manifest = ExperimentManifest(
            experiment_id=self._id_factory(),
            objective=objective,
            context=context,
            created_at=self._clock(),
        )
        self._repository.save(manifest)
        return manifest

    def attach_benchmark_run(
        self, experiment_id: str, run: BenchmarkRunLink
    ) -> ExperimentManifest:
        manifest = self._require(experiment_id)
        updated = replace(manifest, benchmark_runs=manifest.benchmark_runs + (run,))
        self._repository.save(updated)
        return updated

    def attach_telemetry(
        self, experiment_id: str, artifact: ArtifactRef
    ) -> ExperimentManifest:
        manifest = self._require(experiment_id)
        updated = replace(
            manifest, telemetry_artifacts=manifest.telemetry_artifacts + (artifact,)
        )
        self._repository.save(updated)
        return updated

    def attach_profile(
        self, experiment_id: str, artifact: ArtifactRef
    ) -> ExperimentManifest:
        manifest = self._require(experiment_id)
        updated = replace(
            manifest, profiling_artifacts=manifest.profiling_artifacts + (artifact,)
        )
        self._repository.save(updated)
        return updated

    def _require(self, experiment_id: str) -> ExperimentManifest:
        manifest = self._repository.get(experiment_id)
        if manifest is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        return manifest

