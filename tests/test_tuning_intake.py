from coco_agent.conversation.tuning_tools import TuningToolset


def test_partial_goal_is_saved_and_requests_real_context():
    tools = TuningToolset()
    result = tools.update({
        "objective": {
            "primary_metric": "throughput_tokens_per_second",
            "direction": "meet_target",
            "operator": ">=",
            "target_value": 50,
            "unit": "token/s",
        }
    })
    assert result["ready_for_benchmark"] is False
    assert "target.model_ref" in result["missing_fields"]
    assert "workload.input_tokens_min" in result["missing_fields"]


def test_later_turn_merges_sla_and_workload_into_same_task():
    tools = TuningToolset()
    first = tools.update({
        "objective": {"primary_metric": "throughput_tokens_per_second", "target_value": 50}
    })
    second = tools.update({
        "task_id": first["task_id"],
        "constraints": [
            {"metric": "tpot", "operator": "<", "value": 20, "unit": "ms/token", "aggregation": "p95"},
            {"metric": "ttft", "operator": "<", "value": 10, "unit": "s", "aggregation": "p95"},
        ],
        "workload": {"input_tokens_min": 1000, "input_tokens_max": 200000},
    })
    assert second["task_id"] == first["task_id"]
    assert {item["metric"] for item in second["constraints"]} == {"tpot", "ttft"}
    assert second["workload"]["input_tokens_max"] == 200000
    assert "target.model_ref" in second["missing_fields"]
    assert "target.launch_config_ref" in second["missing_fields"]
    assert "target.accelerator_model" in second["missing_fields"]
    assert "target.benchmark_authorized" in second["missing_fields"]


def test_run_gate_never_returns_mock_measurements():
    tools = TuningToolset()
    partial = tools.update({
        "objective": {"primary_metric": "throughput_tokens_per_second", "target_value": 50}
    })
    result = tools.prepare_benchmark({"task_id": partial["task_id"]})
    assert result["status"] == "needs_context"
    assert "report" not in result
    assert "metrics" not in result


def test_complete_intake_reaches_benchmark_adapter_gate():
    tools = TuningToolset()
    task = tools.update({
        "objective": {"primary_metric": "throughput_tokens_per_second", "target_value": 50},
        "workload": {"input_tokens_min": 1000, "input_tokens_max": 200000},
        "target": {
            "model_ref": "local/model", "engine_name": "mindie", "engine_version": "2.3",
            "deployment_ref": "service:primary", "endpoint": "http://127.0.0.1:1025",
            "launch_config_ref": "config/service.json", "accelerator_model": "Ascend 910B",
            "accelerator_count": 8, "benchmark_authorized": True,
        },
    })
    assert task["ready_for_benchmark"] is True
    result = tools.prepare_benchmark({"task_id": task["task_id"]})
    assert result["status"] == "benchmark_plan_ready"
    assert result["execution_state"] == "adapter_unconfigured"
    assert result["adapter"] == "coco_benchmark"
    assert result["configuration_command"] == "coco benchmark config --adapter coco_benchmark"
