"""Minimal OpenAI-compatible chat-completions adapter with tool calling."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import AssistantTurn, ToolCall


class LLMServiceError(RuntimeError):
    """Safe, user-facing model service error without credentials or request bodies."""


class OpenAICompatibleLLM:
    def __init__(self, *, base_url: str, model: str, api_key: str = "") -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleLLM":
        base_url = os.environ.get("LLM_BASE_URL", "")
        model = os.environ.get("LLM_MODEL", "")
        if not base_url or not model:
            raise RuntimeError("请设置 LLM_BASE_URL 和 LLM_MODEL 后再启动真实对话。")
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("LLM_API_KEY", ""),
        )

    @classmethod
    def from_settings(cls, settings) -> "OpenAICompatibleLLM":
        return cls(
            base_url=settings.base_url,
            model=settings.model,
            api_key=settings.api_key,
        )

    def complete(self, messages, tools):
        body = json.dumps({
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(self._url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                error = json.loads(raw).get("error", {})
            except json.JSONDecodeError:
                error = {}
            detail = error.get("message") or exc.reason or "unknown API error"
            code = error.get("code") or error.get("type")
            request_id = exc.headers.get("x-request-id") if exc.headers else None
            suffix = f"；code={code}" if code else ""
            suffix += f"；request_id={request_id}" if request_id else ""
            if exc.code == 429:
                raise LLMServiceError(
                    f"模型服务返回 429：{detail}{suffix}。"
                    "请检查 API 账户额度、Billing 和项目限速后重试。"
                ) from None
            raise LLMServiceError(
                f"OpenAI-compatible API 返回 HTTP {exc.code}：{detail}{suffix}"
            ) from None
        except URLError as exc:
            raise LLMServiceError(f"无法连接模型服务：{exc.reason}") from None
        message = payload["choices"][0]["message"]
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw["function"]
            calls.append(ToolCall(
                id=raw["id"],
                name=function["name"],
                arguments=json.loads(function.get("arguments") or "{}"),
            ))
        return AssistantTurn(message.get("content"), tuple(calls))
