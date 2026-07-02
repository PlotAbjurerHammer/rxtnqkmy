"""LLM abstraction layer: a single async interface over OpenAI-compatible
chat-completions APIs (OpenAI, Anthropic via gateway, vLLM, Ollama, ...),
with retries and token accounting.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, object]] | None = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, object]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse: ...


class OpenAICompatibleLLM:
    """Client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if tools:
            payload["tools"] = tools
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    return self._parse(resp.json())
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"LLM request failed after {self.max_retries} retries: {last_error}")

    def _parse(self, data: dict[str, object]) -> LLMResponse:
        choices = data["choices"]
        assert isinstance(choices, list)
        message = choices[0]["message"]
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                args=json.loads(tc["function"]["arguments"] or "{}"),
            )
            for tc in message.get("tool_calls") or []
        ]
        usage = data.get("usage") or {}
        assert isinstance(usage, dict)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
