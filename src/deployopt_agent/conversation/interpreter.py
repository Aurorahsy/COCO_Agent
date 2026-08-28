"""LLM conversation loop: model -> validated function call -> tool result -> model."""

from __future__ import annotations

import json
from typing import Any

from .models import LLMClient
from .tools import ToolCallError, ToolRegistry


class LLMInterpreter:
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
        skill: str,
        max_turns: int = 8,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._max_turns = max_turns
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": skill}]

    def handle(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        return self.resume()

    def replace_llm(self, llm: LLMClient) -> None:
        self._llm = llm

    def resume(self) -> str:
        """Resume the current logical turn without appending a duplicate user message."""
        for _ in range(self._max_turns):
            turn = self._llm.complete(self.messages, self._tools.definitions())
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.content,
            }
            if turn.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in turn.tool_calls
                ]
            self.messages.append(assistant_message)

            if not turn.tool_calls:
                return turn.content or ""

            for call in turn.tool_calls:
                try:
                    result = self._tools.execute(call.name, call.arguments)
                    payload = {"ok": True, "result": result}
                except (ToolCallError, ValueError) as exc:
                    payload = {"ok": False, "error": str(exc)}
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(payload, ensure_ascii=False),
                })
        raise RuntimeError(f"LLM tool loop exceeded {self._max_turns} turns")
