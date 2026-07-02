"""Tool system: register plain Python functions as tools via the @tool
decorator. JSON schemas are generated from type annotations and docstrings.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from agentkit.events import Observation

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, object]
    fn: Callable[..., object]

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(fn: Callable[..., object], name: str | None = None) -> Tool:
    """Build a Tool from a function's signature and docstring."""
    sig = inspect.signature(fn)
    properties: dict[str, object] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        json_type = _TYPE_MAP.get(annotation, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    return Tool(
        name=name or fn.__name__,
        description=inspect.getdoc(fn) or "",
        parameters={"type": "object", "properties": properties, "required": required},
        fn=fn,
    )


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"tool already registered: {t.name}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, object]]:
        return [t.schema() for t in self._tools.values()]

    async def execute(self, name: str, args: dict[str, object]) -> Observation:
        t = self._tools.get(name)
        if t is None:
            return Observation(content=f"unknown tool: {name}", is_error=True)
        try:
            result = t.fn(**args)
            if inspect.isawaitable(result):
                result = await result
            return Observation(content=str(result))
        except Exception as exc:  # noqa: BLE001 - tool failures become observations
            return Observation(content=f"{type(exc).__name__}: {exc}", is_error=True)
