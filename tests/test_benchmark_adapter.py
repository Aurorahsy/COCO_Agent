from pathlib import Path

from coco_agent import configuration
from coco_agent.benchmark import (
    BenchmarkSettings,
    LocalCocoBenchmarkAdapter,
    load_benchmark_settings,
    save_benchmark_settings,
)
from coco_agent.benchmark.runner import ProcessResult


class FakeRunner:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    def run(self, command, *, cwd, environment=None):
        self.calls.append((command, cwd, environment))
        return ProcessResult(self.returncode, '{"ok": true}', "")


class FakeDatasetRunner(FakeRunner):
    def run(self, command, *, cwd, environment=None):
        self.calls.append((command, cwd, environment))
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "requests.jsonl").write_text("{}\n", encoding="utf-8")
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return ProcessResult(0, '{"requests": 1}', "")


def settings(tmp_path):
    return BenchmarkSettings(
        executable=str(tmp_path / "coco-benchmark.exe"),
        workdir=str(tmp_path / "benchmark"),
        output_dir=str(tmp_path / "results"),
        api_key_env="DEEPSEEK_API_KEY",
    )


def test_benchmark_configuration_is_external_and_contains_no_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    value = settings(tmp_path)
    path = save_benchmark_settings(value)
    assert path.parent == configuration.config_path().parent
    assert path.name == "benchmark.json"
    assert "DEEPSEEK_API_KEY" in path.read_text(encoding="utf-8")
    assert load_benchmark_settings() == value


def test_local_adapter_prepares_coco_benchmark_and_normalizes_base_endpoint(tmp_path):
    adapter = LocalCocoBenchmarkAdapter(settings(tmp_path))
    task = {
        "target": {
            "endpoint": "https://api.deepseek.com/v1/chat/completions",
            "model_ref": "deepseek-chat",
            "engine_name": "vLLM",
        },
        "workload": {"input_tokens_min": 512, "input_tokens_max": 10240},
    }
    result = adapter.prepare("task-1", task)
    assert result["adapter"] == "coco_benchmark"
    assert result["execution_state"] == "awaiting_confirmation"
    assert result["plan"]["endpoint"] == "https://api.deepseek.com"
    assert result["plan"]["workload_generation_required"] is True
    assert result["ui_event"]["adapter"] == "COCO_Benchmark"


def test_local_adapter_executes_confirmed_plan_without_exposing_secret(tmp_path, monkeypatch):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    workload = benchmark_dir / "requests.jsonl"
    workload.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    runner = FakeRunner()
    adapter = LocalCocoBenchmarkAdapter(settings(tmp_path), runner=runner)
    task = {
        "target": {
            "endpoint": "https://api.deepseek.com/chat/completions",
            "model_ref": "deepseek-chat",
            "engine_name": "vllm",
        },
        "workload": {"workload_ref": "requests.jsonl"},
    }
    plan = adapter.prepare("tuning-1", task)["plan"]
    result = adapter.execute("tuning-1", plan)
    command, cwd, environment = runner.calls[0]
    assert result["execution_state"] == "completed"
    assert cwd == str(benchmark_dir)
    assert "--api-key-env" in command
    assert "DEEPSEEK_API_KEY" in command
    assert "secret-value" not in command
    assert environment == {"DEEPSEEK_API_KEY": "secret-value"}


def test_workload_is_materialized_without_endpoint_or_credential(tmp_path):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    runner = FakeDatasetRunner()
    adapter = LocalCocoBenchmarkAdapter(settings(tmp_path), runner=runner)
    result = adapter.generate_workload("tuning-offline", {
        "target": {"model_ref": "model-ref"},
        "workload": {
            "input_tokens_min": 1000,
            "input_tokens_max": 10000,
            "output_tokens_max": 512,
            "request_count": 100,
            "arrival_pattern": "poisson",
        },
    })
    assert result["status"] == "workload_ready"
    assert result["request_count"] == 100
    assert result["dataset_shape"] == {"chains": 10, "turns_per_chain": 10}
    assert Path(result["workload_ref"]).is_file()
    command, _, environment = runner.calls[0]
    assert "build-dataset" in command
    assert environment is None
