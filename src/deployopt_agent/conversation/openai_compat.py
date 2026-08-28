"""Minimal OpenAI-compatible chat-completions adapter with tool calling."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import AssistantTurn, ToolCall


class LLMServiceError(RuntimeError):
    """Safe, user-facing model service error without credentials or request bodies."""


class AuthenticationFailure(LLMServiceError):
    """The endpoint rejected the configured credential."""


class EndpointOrModelFailure(LLMServiceError):
    """The endpoint or configured model does not exist."""


class RateLimited(LLMServiceError):
    """The provider rejected the request due to quota or rate limiting."""


class ServiceUnavailable(LLMServiceError):
    """The provider or network is temporarily unavailable."""


class ServiceTimeout(LLMServiceError):
    """The provider did not respond within the configured timeout."""


class InvalidServiceResponse(LLMServiceError):
    """The provider returned a response that does not match the expected schema."""


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

    def _request(self, messages, tools=None):
        body = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": 0,
        } | ({"tools": tools, "tool_choice": "auto"} if tools else {})).encode("utf-8")
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
            if exc.code in {401, 403}:
                raise AuthenticationFailure(
                    f"模型服务认证失败（HTTP {exc.code}）：{detail}{suffix}。"
                    "请重新检查 API Key。"
                ) from None
            if exc.code == 404:
                raise EndpointOrModelFailure(
                    f"模型服务地址或模型不存在（HTTP 404）：{detail}{suffix}。"
                    "请检查 Base URL 和模型名称。"
                ) from None
            if exc.code == 429:
                raise RateLimited(
                    f"模型服务返回 429：{detail}{suffix}。"
                    "请检查 API 账户额度、Billing 和项目限速后重试。"
                ) from None
            if exc.code in {408, 504}:
                raise ServiceTimeout(
                    f"模型服务请求超时（HTTP {exc.code}）：{detail}{suffix}"
                ) from None
            if 500 <= exc.code < 600:
                raise ServiceUnavailable(
                    f"模型服务暂时不可用（HTTP {exc.code}）：{detail}{suffix}"
                ) from None
            raise LLMServiceError(
                f"OpenAI-compatible API 返回 HTTP {exc.code}：{detail}{suffix}"
            ) from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ServiceTimeout("连接模型服务超时，请检查网络或服务地址。") from None
            raise ServiceUnavailable(f"无法连接模型服务：{exc.reason}") from None
        except TimeoutError:
            raise ServiceTimeout("模型服务响应超时，请稍后重试。") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise InvalidServiceResponse("模型服务返回了无法识别的响应格式。") from None
        try:
            return payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidServiceResponse("模型服务返回了无法识别的响应格式。") from exc

    def probe(self) -> None:
        """Validate endpoint, credential and model without enabling tool calls."""
        self._request([{"role": "user", "content": "Reply with OK."}])

    def complete(self, messages, tools):
        message = self._request(messages, tools)
        calls = []
        try:
            for raw in message.get("tool_calls") or []:
                function = raw["function"]
                calls.append(ToolCall(
                    id=raw["id"],
                    name=function["name"],
                    arguments=json.loads(function.get("arguments") or "{}"),
                ))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidServiceResponse("模型服务返回了无效的 Function Call。") from exc
        return AssistantTurn(message.get("content"), tuple(calls))
