"""Regression for the multi-turn intake reported during manual CLI testing."""

from coco_agent.conversation.tuning_tools import TuningToolset


def test_deepseek_vllm_conversation_waits_for_explicit_facts_and_authorization():
    tools = TuningToolset()
    task = tools.update({"objective": {
        "description": "总吞吐量至少 50 token/s",
        "primary_metric": "throughput_tokens_per_second",
        "direction": "meet_target", "operator": ">=", "target_value": 50,
        "unit": "token/s",
    }})
    task_id = task["task_id"]

    task = tools.update({"task_id": task_id, "workload": {
        "input_tokens_min": 512, "input_tokens_max": 10240,
        "output_tokens_min": 1, "output_tokens_max": 1024,
    }})
    task = tools.update({"task_id": task_id, "constraints": [
        {"metric": "ttft", "operator": "<=", "value": 2, "unit": "s"},
        {"metric": "tpot", "operator": "<=", "value": 20, "unit": "ms/token"},
    ]})
    task = tools.update({"task_id": task_id, "target": {
        "model_ref": "DeepSeek V4", "engine_name": "vLLM",
        "engine_version": "0.23.0", "deployment_ref": "cloud",
        "endpoint": "[https://api.deepseek.com/chat/completions](https://api.deepseek.com/chat/completions)",
    }})

    assert task["target"]["endpoint"] == "https://api.deepseek.com/chat/completions"
    assert "constraints[0].aggregation" in task["missing_fields"]
    assert "constraints[1].aggregation" in task["missing_fields"]
    assert "target.launch_config_ref" in task["missing_fields"]
    assert "target.accelerator_model" in task["missing_fields"]
    assert "target.accelerator_count" in task["missing_fields"]
    assert "target.benchmark_authorized" in task["missing_fields"]
    assert tools.prepare_benchmark({"task_id": task_id})["status"] == "needs_context"

    task = tools.update({"task_id": task_id, "constraints": [
        {"metric": "ttft", "operator": "<=", "value": 2, "unit": "s", "aggregation": "p95"},
        {"metric": "tpot", "operator": "<=", "value": 20, "unit": "ms/token", "aggregation": "p95"},
    ], "target": {
        "launch_config_ref": "deployment/config.json", "accelerator_model": "accelerator",
        "accelerator_count": 8, "benchmark_authorized": True,
    }})
    assert task["ready_for_benchmark"] is True
    plan = tools.prepare_benchmark({"task_id": task_id})
    assert plan["status"] == "benchmark_plan_ready"
    assert plan["execution_state"] == "adapter_unconfigured"
    assert plan["ui_event"]["kind"] == "benchmark_plan"


def test_confirmation_tools_are_exposed_to_the_model():
    names = {
        item["function"]["name"]
        for item in TuningToolset().registry().definitions()
    }
    assert {
        "generate_benchmark_workload",
        "confirm_benchmark_run",
        "cancel_benchmark_run",
    } <= names


def test_workload_generation_does_not_require_endpoint_or_authorization():
    class WorkloadGenerator:
        def generate_workload(self, task_id, task):
            return {
                "task_id": task_id,
                "status": "workload_ready",
                "execution_state": "completed",
                "workload_ref": "D:/results/requests.jsonl",
            }

    tools = TuningToolset(WorkloadGenerator())
    task = tools.update({
        "workload": {
            "input_tokens_min": 1000,
            "input_tokens_max": 10000,
            "output_tokens_max": 512,
            "arrival_pattern": "burst",
        },
        "target": {"model_ref": "model-or-weight-ref"},
    })
    assert task["ready_for_workload"] is True
    assert task["ready_for_benchmark"] is False
    result = tools.generate_workload({"task_id": task["task_id"]})
    assert result["status"] == "workload_ready"
    assert tools._tasks[task["task_id"]]["workload"]["workload_ref"].endswith("requests.jsonl")
