from __future__ import annotations

import json

import pytest

from coco_agent.cli import run_chat
from coco_agent.conversation.interpreter import LLMInterpreter
from coco_agent.conversation.models import AssistantTurn, ToolCall
from coco_agent.conversation.openai_compat import ServiceTimeout
from coco_agent.conversation.tools import RegisteredTool, ToolRegistry


class TuningScriptedLLM:
    """Model test double; the interpreter still performs the real tool loop."""

    def __init__(self):
        self.calls = 0
        self.received_tools = []
        self.seen_messages = []

    def complete(self, messages, tools):
        self.calls += 1
        self.received_tools = tools
        self.seen_messages = list(messages)
        if self.calls == 1:
            return AssistantTurn(tool_calls=(ToolCall(
                "call-submit",
                "update_tuning_task",
                {
                    "objective": {
                        "description": "把吞吐量优化到 100 token/s",
                        "primary_metric": "throughput_tokens_per_second",
                        "direction": "meet_target",
                        "operator": ">=",
                        "target_value": 100,
                        "unit": "token/s",
                    },
                    "workload": {"input_tokens_min": 1000, "input_tokens_max": 200000},
                },
            ),))
        if self.calls == 2:
            submit_result = json.loads(messages[-1]["content"])
            task_id = submit_result["result"]["task_id"]
            missing = submit_result["result"]["missing_fields"]
            return AssistantTurn(f"已记录目标，还需要：{', '.join(missing)}。任务 {task_id} 尚未执行。")


def test_cli_executes_llm_function_call_chain():
    llm = TuningScriptedLLM()
    answers = iter(["帮我把吞吐量优化到 100", "exit"])
    output = []
    assert run_chat(llm, lambda _prompt: next(answers), output.append) == 0
    assert llm.calls == 2
    assert {item["function"]["name"] for item in llm.received_tools} == {
        "inspect_benchmark_capabilities",
        "update_tuning_task",
        "generate_benchmark_workload",
        "prepare_benchmark_run",
        "confirm_benchmark_run",
        "cancel_benchmark_run",
    }
    assert "尚未执行" in "\n".join(output)
    tool_messages = [item for item in llm.seen_messages if item["role"] == "tool"]
    assert [item["name"] for item in tool_messages] == [
        "update_tuning_task",
    ]


def test_cli_uses_minimal_prompt_and_displays_model_reasoning():
    class ReasoningLLM:
        def complete(self, messages, tools):
            return AssistantTurn("可以开始。", reasoning="先确认用户目标。")

    prompts = []
    answers = iter(["开始", "/exit"])
    output = []

    assert run_chat(
        ReasoningLLM(),
        lambda prompt: prompts.append(prompt) or next(answers),
        output.append,
    ) == 0
    assert prompts == ["❯ ", "❯ "]
    rendered = "\n".join(output)
    assert "You>" not in rendered
    assert "Agent>" not in rendered
    assert "思考  先确认用户目标。" in rendered
    assert "● coco\n  可以开始。" in rendered


class InvalidThenRecoveringLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return AssistantTurn(tool_calls=(ToolCall(
                "bad", "example", {"count": "not-an-integer"}
            ),))
        error = json.loads(messages[-1]["content"])
        assert error["ok"] is False
        return AssistantTurn("参数无效，请补充正确的信息。")


def test_invalid_tool_arguments_are_returned_to_model_not_executed():
    executed = []
    registry = ToolRegistry([RegisteredTool(
        "example",
        "example tool",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
        lambda arguments: executed.append(arguments) or {"done": True},
    )])
    interpreter = LLMInterpreter(
        llm=InvalidThenRecoveringLLM(), tools=registry, skill="test"
    )
    assert interpreter.handle("run") == "参数无效，请补充正确的信息。"
    assert executed == []


def test_unknown_tool_is_reported_to_model():
    class UnknownToolLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return AssistantTurn(tool_calls=(ToolCall("x", "delete_system", {}),))
            assert "unknown tool" in messages[-1]["content"]
            return AssistantTurn("无法执行该工具。")

    interpreter = LLMInterpreter(
        llm=UnknownToolLLM(), tools=ToolRegistry([]), skill="test"
    )
    assert interpreter.handle("do something") == "无法执行该工具。"


def test_loop_has_hard_turn_limit():
    class EndlessLLM:
        def complete(self, messages, tools):
            return AssistantTurn(tool_calls=(ToolCall("x", "missing", {}),))

    interpreter = LLMInterpreter(
        llm=EndlessLLM(), tools=ToolRegistry([]), skill="test", max_turns=2
    )
    with pytest.raises(RuntimeError, match="exceeded 2 turns"):
        interpreter.handle("loop")


def test_cli_retries_timeout_without_duplicating_user_message():
    class TimeoutThenSuccessLLM:
        def __init__(self):
            self.calls = 0
            self.seen_messages = []

        def complete(self, messages, tools):
            self.calls += 1
            self.seen_messages = list(messages)
            if self.calls == 1:
                raise ServiceTimeout("模型服务响应超时，请稍后重试。")
            return AssistantTurn("恢复成功。")

    llm = TimeoutThenSuccessLLM()
    answers = iter(["你好", "r", "exit"])
    output = []
    assert run_chat(llm, lambda _prompt: next(answers), output.append) == 0
    assert llm.calls == 2
    assert [message["content"] for message in llm.seen_messages if message["role"] == "user"] == ["你好"]
    assert "恢复成功" in "\n".join(output)


def test_configuration_is_saved_only_after_validation(tmp_path, monkeypatch):
    from coco_agent import configuration
    from coco_agent.cli import configure

    monkeypatch.setenv("APPDATA", str(tmp_path))
    answers = iter(["https://example.com/v1", "model"])

    def reject(_settings):
        raise ServiceTimeout("validation timeout")

    with pytest.raises(ServiceTimeout):
        configure(
            input_fn=lambda _prompt: next(answers),
            secret_fn=lambda _prompt: "secret",
            output_fn=lambda _message: None,
            validator=reject,
        )
    assert not configuration.config_path().exists()
