from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

from deployopt_agent.conversation.openai_compat import (
    LLMServiceError,
    OpenAICompatibleLLM,
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
