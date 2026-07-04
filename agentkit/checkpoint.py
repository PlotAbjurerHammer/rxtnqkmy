"""Checkpointing: persist agent run state (messages + step counter) to JSON
so an interrupted run can resume from where it left off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agentkit.llm import Message


@dataclass
class RunState:
    task: str
    messages: list[Message]
    step: int = 0
    total_tokens: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


class Checkpointer:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: RunState) -> None:
        payload = {
            "task": state.task,
            "step": state.step,
            "total_tokens": state.total_tokens,
            "metadata": state.metadata,
            "messages": [self._message_to_dict(m) for m in state.messages],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> RunState | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return RunState(
            task=payload["task"],
            step=payload["step"],
            total_tokens=payload.get("total_tokens", 0),
            metadata=payload.get("metadata", {}),
            messages=[Message(**m) for m in payload["messages"]],
        )

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    @staticmethod
    def _message_to_dict(m: Message) -> dict[str, object]:
        return {
            "role": m.role,
            "content": m.content,
            "tool_call_id": m.tool_call_id,
            "name": m.name,
            "tool_calls": m.tool_calls,
        }
