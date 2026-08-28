"""Safe mock capability backed by a durable idempotency store."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from ..domain.contracts import ActionRequest, ActionResult, GoalSpec


class SqliteOperationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with closing(self._connect()) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    result_json TEXT NOT NULL
                )"""
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get_by_operation(self, operation_id: str) -> ActionResult | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT result_json FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return self._decode(row[0]) if row else None

    def get_by_key(self, idempotency_key: str) -> ActionResult | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT result_json FROM operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return self._decode(row[0]) if row else None

    def put(self, request: ActionRequest, result: ActionResult) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO operations(operation_id, idempotency_key, result_json) VALUES (?, ?, ?)",
                (request.operation_id, request.idempotency_key, json.dumps(result.to_dict())),
            )
            conn.commit()

    @staticmethod
    def _decode(raw: str) -> ActionResult:
        return ActionResult(**json.loads(raw))


class MockTuningCapability:
    """Deterministic capability; it never executes host commands."""

    def __init__(self, store: SqliteOperationStore) -> None:
        self.store = store
        self.execution_count = 0

    def inspect_environment(self, goal: GoalSpec) -> dict[str, Any]:
        return {
            "hardware": goal.inputs.get("hardware", "mock-gpu"),
            "model": goal.inputs.get("model", "mock-model"),
            "runtime": "mock-runtime-v1",
        }

    def execute(self, request: ActionRequest) -> ActionResult:
        existing = self.store.get_by_key(request.idempotency_key)
        if existing is not None:
            return ActionResult(
                operation_id=existing.operation_id,
                success=existing.success,
                output=existing.output,
                reused=True,
                error_code=existing.error_code,
            )

        self.execution_count += 1
        output = dict(request.arguments.get("mock_metrics", {"throughput": 120.0}))
        output["tested_config"] = dict(request.arguments.get("config", {}))
        output["benchmark_phase"] = request.arguments.get("phase", "candidate")
        result = ActionResult(request.operation_id, True, output)
        self.store.put(request, result)
        return result

    def status(self, operation_id: str) -> ActionResult | None:
        return self.store.get_by_operation(operation_id)
