from __future__ import annotations

import json

from deployopt_agent import configuration
from deployopt_agent.cli import configure, run_chat, show_configuration
from deployopt_agent.conversation.models import AssistantTurn
from deployopt_agent.configuration import LLMSettings, load_settings, save_settings


def fake_dpapi(monkeypatch):
    monkeypatch.setattr(configuration, "_protect", lambda value: f"encrypted:{value[::-1]}")
    monkeypatch.setattr(
        configuration, "_unprotect", lambda value: value.removeprefix("encrypted:")[::-1]
    )


def test_single_external_config_is_encrypted_and_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    fake_dpapi(monkeypatch)
    secret = "test-credential-value"
    first = save_settings(LLMSettings("https://api.openai.com/v1", "model-a", secret))
    second = save_settings(LLMSettings("https://api.openai.com/v1", "model-b", secret))
    assert first == second
    assert list(first.parent.glob("*.json")) == [first]
    raw = first.read_text(encoding="utf-8")
    assert secret not in raw
    assert json.loads(raw)["model"] == "model-b"
    assert load_settings().api_key == secret


def test_config_cli_never_prints_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    fake_dpapi(monkeypatch)
    output = []
    answers = iter(["", ""])
    secret = "test-credential-never-log"
    assert configure(lambda _prompt: next(answers), lambda _prompt: secret, output.append) == 0
    assert secret not in "\n".join(output)
    shown = []
    assert show_configuration(shown.append) == 0
    assert secret not in "\n".join(shown)
    assert "DPAPI" in "\n".join(shown)


def test_first_chat_launch_configures_once_then_reuses(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    fake_dpapi(monkeypatch)

    class GreetingLLM:
        def complete(self, messages, tools):
            return AssistantTurn("你好，我已经准备好了。")

    monkeypatch.setattr(
        "deployopt_agent.cli.OpenAICompatibleLLM.from_settings",
        lambda _settings: GreetingLLM(),
    )
    first_answers = iter([
        "https://api.deepseek.com",
        "deepseek-chat",
        "你好",
        "exit",
    ])
    first_output = []
    assert run_chat(
        input_fn=lambda _prompt: next(first_answers),
        output_fn=first_output.append,
        secret_fn=lambda _prompt: "test-first-launch-credential",
    ) == 0
    assert "首次启动" in "\n".join(first_output)
    assert configuration.config_path().is_file()

    second_answers = iter(["exit"])
    second_output = []
    assert run_chat(
        input_fn=lambda _prompt: next(second_answers),
        output_fn=second_output.append,
        secret_fn=lambda _prompt: (_ for _ in ()).throw(AssertionError("must not prompt")),
    ) == 0
    assert "首次启动" not in "\n".join(second_output)
    assert "已加载持久化模型配置" in "\n".join(second_output)


def test_legacy_config_is_moved_to_single_coco_agent_location(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    fake_dpapi(monkeypatch)
    legacy = tmp_path / "DeployOpt Agent" / "config.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps({
            "version": 1,
            "provider": "openai_compatible",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "api_key_dpapi": "encrypted:yek",
        }),
        encoding="utf-8",
    )
    settings = load_settings()
    canonical = tmp_path / "coco_agent" / "config.json"
    assert settings.api_key == "key"
    assert canonical.is_file()
    assert not legacy.exists()
    assert len(list(tmp_path.rglob("config.json"))) == 1
