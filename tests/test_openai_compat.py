from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from deployopt_agent.conversation.openai_compat import (
    LLMServiceError,
    OpenAICompatibleLLM,
    ServiceTimeout,
)


def test_429_is_converted_to_actionable_safe_error(monkeypatch):
    body = BytesIO(
        b'{"error":{"message":"quota exhausted","type":"insufficient_quota",'
        b'"code":"insufficient_quota"}}'
    )
    headers = Message()
    headers["x-request-id"] = "req-test"

    def fail(*_args, **_kwargs):
        raise HTTPError(
            "https://api.openai.com/v1/chat/completions",
            429,
            "Too Many Requests",
            headers,
            body,
        )

    monkeypatch.setattr("deployopt_agent.conversation.openai_compat.urlopen", fail)
    client = OpenAICompatibleLLM(
        base_url="https://api.openai.com/v1",
        model="test-model",
        api_key="test-credential-never-print",
    )
    with pytest.raises(LLMServiceError) as captured:
        client.complete([{"role": "user", "content": "hello"}], [])
    message = str(captured.value)
    assert "429" in message
    assert "insufficient_quota" in message
    assert "req-test" in message
    assert "test-credential-never-print" not in message


def test_504_is_classified_as_recoverable_timeout(monkeypatch):
    def fail(*_args, **_kwargs):
        raise HTTPError(
            "https://example.com/v1/chat/completions",
            504,
            "Gateway Time-out",
            Message(),
            BytesIO(b"Gateway Time-out"),
        )

    monkeypatch.setattr("deployopt_agent.conversation.openai_compat.urlopen", fail)
    client = OpenAICompatibleLLM(
        base_url="https://example.com/v1", model="model", api_key="secret"
    )
    with pytest.raises(ServiceTimeout, match="504"):
        client.complete([{"role": "user", "content": "hello"}], [])


def test_network_timeout_is_classified_without_leaking_secret(monkeypatch):
    def fail(*_args, **_kwargs):
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr("deployopt_agent.conversation.openai_compat.urlopen", fail)
    client = OpenAICompatibleLLM(
        base_url="https://example.com/v1", model="model", api_key="secret-value"
    )
    with pytest.raises(ServiceTimeout) as captured:
        client.probe()
    assert "secret-value" not in str(captured.value)


def test_non_json_response_is_classified(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr(
        "deployopt_agent.conversation.openai_compat.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    client = OpenAICompatibleLLM(
        base_url="https://example.com/v1", model="model", api_key="secret"
    )
    with pytest.raises(LLMServiceError, match="无法识别"):
        client.probe()
