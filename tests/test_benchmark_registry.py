from coco_agent.benchmark import (
    BenchmarkAdapterRegistry,
    BenchmarkSettings,
    LocalAisBenchAdapter,
    LocalCocoBenchmarkAdapter,
)


def settings(adapter):
    executable = "coco-benchmark" if adapter == "coco_benchmark" else "ais_bench"
    return BenchmarkSettings(executable, "/work", "/results", "TARGET_API_KEY", adapter)


def task(preferred="auto"):
    return {
        "benchmark": {"preferred_adapter": preferred},
        "target": {
            "endpoint": "http://target/v1/chat/completions",
            "model_ref": "model",
            "engine_name": "vllm",
        },
        "workload": {"workload_ref": "requests.jsonl"},
    }


def test_auto_prefers_coco_benchmark_when_both_are_configured():
    registry = BenchmarkAdapterRegistry()
    registry.register(LocalAisBenchAdapter(settings("ais_bench")))
    registry.register(LocalCocoBenchmarkAdapter(settings("coco_benchmark")))
    assert registry.prepare("task", task())["adapter"] == "coco_benchmark"


def test_explicit_aisbench_selection_uses_aisbench_contract():
    registry = BenchmarkAdapterRegistry()
    registry.register(LocalCocoBenchmarkAdapter(settings("coco_benchmark")))
    registry.register(LocalAisBenchAdapter(settings("ais_bench")))
    value = task("ais_bench")
    value["benchmark"].update({
        "ais_model_profile": "vllm_api_general_stream",
        "ais_dataset_profile": "synthetic_gen",
    })
    result = registry.prepare("task", value)
    assert result["adapter"] == "ais_bench"
    assert result["plan"]["command"][-1] == "perf"


def test_unconfigured_explicit_adapter_returns_its_configuration_command():
    registry = BenchmarkAdapterRegistry()
    result = registry.prepare("task", task("ais_bench"))
    assert result["adapter"] == "ais_bench"
    assert result["configuration_command"].endswith("--adapter ais_bench")
    catalog = {item["adapter"]: item for item in result["available_adapters"]}
    assert "accuracy_report" in catalog["ais_bench"]["evidence_features"]
    assert "run_receipt" in catalog["coco_benchmark"]["evidence_features"]


def test_coco_supports_p95_tpot_as_a_derived_metric():
    registry = BenchmarkAdapterRegistry()
    registry.register(LocalCocoBenchmarkAdapter(settings("coco_benchmark")))
    registry.register(LocalAisBenchAdapter(settings("ais_bench")))
    value = task("coco_benchmark")
    value["objective"] = {"primary_metric": "throughput_tokens_per_second"}
    value["constraints"] = [
        {"metric": "ttft", "aggregation": "p95"},
        {"metric": "tpot", "aggregation": "p95"},
    ]
    result = registry.prepare("task", value)
    assert result["capability_match"]["complete"] is True
    assert result["capability_match"]["gaps"] == []
    catalog = {item["adapter"]: item for item in result["available_adapters"]}
    assert catalog["coco_benchmark"]["derived_metrics"] == (
        "tpot=(e2e_ms-ttft_ms)/(output_tokens-1)",
    )


def test_confirmation_does_not_start_when_workload_needs_generation():
    registry = BenchmarkAdapterRegistry()
    registry.register(LocalCocoBenchmarkAdapter(settings("coco_benchmark")))
    value = task()
    value["workload"] = {}
    registry.prepare("needs-workload", value)
    result = registry.confirm("needs-workload")
    assert result["status"] == "needs_workload"
    assert result["execution_state"] == "not_started"
