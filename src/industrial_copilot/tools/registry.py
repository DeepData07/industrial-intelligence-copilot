"""Strict registry: only declared tool names and Pydantic argument schemas can execute."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from industrial_copilot.copilot.schemas import ToolCall


class UnknownToolError(ValueError):
    """Raised when a planner requests a tool outside the fixed registry."""


class ToolArgumentError(ValueError):
    """Raised when a known tool receives invalid structured arguments."""


ToolExecutor = Callable[[BaseModel], Any]


@dataclass(frozen=True)
class RegisteredTool:
    """Name, description, validated argument model, and deterministic executor."""

    name: str
    description: str
    arguments_model: type[BaseModel]
    executor: ToolExecutor


class ToolRegistry:
    """Registry that prevents arbitrary Python, SQL, function names, or parameter shapes."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Register one unique tool during application startup."""

        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def execute(self, call: ToolCall) -> Any:
        """Validate a tool call's name and arguments before executing its fixed callable."""

        if call.name not in self._tools:
            raise UnknownToolError(f"Unknown tool: {call.name}")
        tool = self._tools[call.name]
        try:
            arguments = tool.arguments_model.model_validate(call.arguments)
        except Exception as error:
            raise ToolArgumentError(f"Invalid arguments for {call.name}: {error}") from error
        return tool.executor(arguments)

    def describe(self) -> list[dict[str, str]]:
        """Return compact planner-facing tool descriptions, without executable implementation detail."""

        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def describe_with_schemas(self) -> list[dict[str, object]]:
        """Expose a non-executable tool catalogue for a bounded planner."""

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "arguments_schema": tool.arguments_model.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    @property
    def names(self) -> frozenset[str]:
        """Expose the immutable whitelist for plan validation and evaluation."""

        return frozenset(self._tools)
