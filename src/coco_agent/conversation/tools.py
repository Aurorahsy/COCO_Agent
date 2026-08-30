"""Validated tool registry exposed to the LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ToolCallError(ValueError):
    pass


def _validate(schema: dict[str, Any], value: Any, path: str = "arguments") -> None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise ToolCallError(f"{path} must be an object")
        properties = schema.get("properties", {})
        unknown = set(value) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ToolCallError(f"{path} contains unknown fields: {sorted(unknown)}")
        for name in schema.get("required", []):
            if name not in value:
                raise ToolCallError(f"{path}.{name} is required")
        for name, item in value.items():
            if name in properties:
                _validate(properties[name], item, f"{path}.{name}")
    elif expected == "string" and not isinstance(value, str):
        raise ToolCallError(f"{path} must be a string")
    elif expected == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ToolCallError(f"{path} must be a number")
    elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ToolCallError(f"{path} must be an integer")
    elif expected == "boolean" and not isinstance(value, bool):
        raise ToolCallError(f"{path} must be a boolean")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolCallError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and "exclusiveMinimum" in schema:
        if value <= schema["exclusiveMinimum"]:
            raise ToolCallError(f"{path} must be greater than {schema['exclusiveMinimum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema:
        if value < schema["minimum"]:
            raise ToolCallError(f"{path} must be at least {schema['minimum']}")


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[RegisteredTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolCallError(f"unknown tool: {name}")
        _validate(tool.parameters, arguments)
        return tool.handler(arguments)
