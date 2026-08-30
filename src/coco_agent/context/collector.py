"""Composition service for independently implemented context inspectors."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Callable

from ..domain.tuning import PerformanceObjective, SystemContextSnapshot
from ..domain.tuning_ports import (
    DeploymentInspector,
    EngineInspector,
    HardwareInspector,
    ModelInspector,
)


class CompositeContextCollector:
    def __init__(
        self,
        *,
        model: ModelInspector,
        engine: EngineInspector,
        deployment: DeploymentInspector,
        hardware: HardwareInspector,
        id_factory: Callable[[], str],
        clock: Callable[[], str],
    ) -> None:
        self._model = model
        self._engine = engine
        self._deployment = deployment
        self._hardware = hardware
        self._id_factory = id_factory
        self._clock = clock

    def collect(self, objective: PerformanceObjective) -> SystemContextSnapshot:
        target = objective.target
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="coco-context") as executor:
            model_future = executor.submit(self._model.inspect, target.model_ref)
            engine_future = executor.submit(self._engine.inspect, target.deployment_ref)
            deployment_future = executor.submit(
                self._deployment.inspect, target.deployment_ref
            )
            hardware_future = executor.submit(self._hardware.inspect, target.deployment_ref)
            model = model_future.result()
            engine = engine_future.result()
            deployment = deployment_future.result()
            hardware = hardware_future.result()

        facts = {
            "model": self._without_provenance(asdict(model)),
            "engine": self._without_provenance(asdict(engine)),
            "deployment": self._without_provenance(asdict(deployment)),
            "hardware": self._without_provenance(asdict(hardware)),
        }
        canonical = json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return SystemContextSnapshot(
            snapshot_id=self._id_factory(),
            captured_at=self._clock(),
            model=model,
            engine=engine,
            deployment=deployment,
            hardware=hardware,
            fingerprint=fingerprint,
        )

    @classmethod
    def _without_provenance(cls, value):
        if isinstance(value, dict):
            return {
                key: cls._without_provenance(item)
                for key, item in value.items()
                if key != "provenance"
            }
        if isinstance(value, (list, tuple)):
            return [cls._without_provenance(item) for item in value]
        return value

