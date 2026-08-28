"""Run the smallest coco_agent tuning chain with deterministic mock benchmarks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from coco_agent.capabilities import MockTuningCapability, SqliteOperationStore
from coco_agent.domain.contracts import Criterion, GoalSpec
from coco_agent.demo import (
    AllowLowRiskActions,
    DoubleConcurrencyPolicy,
    InMemoryExperience,
    NumericGoalVerifier,
    SimpleReportBuilder,
)
from coco_agent.graph_app import TuningApplication


def main() -> None:
    goal = GoalSpec(
        task_id="minimal-chain-demo",
        objective="increase throughput to at least 100 requests/s",
        acceptance_criteria=(Criterion("throughput", ">=", 100.0),),
        inputs={
            "hardware": "mock-a100-80gb",
            "model": "mock-7b-model",
            "baseline_config": {"max_num_seqs": 128},
            "baseline_metrics": {"throughput": 80.0, "latency_p99_ms": 220.0},
            "candidate_metrics": {"throughput": 120.0, "latency_p99_ms": 170.0},
        },
    )

    with tempfile.TemporaryDirectory(prefix="deployopt_minimal_") as temp_dir:
        root = Path(temp_dir)
        with SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite")) as checkpointer:
            app = TuningApplication(
                checkpointer=checkpointer,
                capability=MockTuningCapability(
                    SqliteOperationStore(root / "operations.sqlite")
                ),
                authorization=AllowLowRiskActions(),
                verifier=NumericGoalVerifier(),
                experience=InMemoryExperience(),
                tuning_policy=DoubleConcurrencyPolicy(),
                report_builder=SimpleReportBuilder(),
            )
            result = app.start(goal)

    output = {
        "task_id": result["task_id"],
        "status": result["status"],
        "environment": result["environment"],
        "analysis": result["metric_analysis"],
        "recommendation": result["recommendation"],
        "comparison_report": result["terminal_result"]["comparison_report"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
