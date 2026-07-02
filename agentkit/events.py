"""Event stream: every step of an agent run is recorded as an Action or
Observation, appended to a JSONL trajectory file and broadcast to subscribers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Action:
    """The agent's intent to call a tool, with its reasoning."""

    tool_name: str
    args: dict[str, object]
    thought: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def kind(self) -> str:
        return "action"


@dataclass
class Observation:
    """The result of executing an Action."""

    content: str
    is_error: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def kind(self) -> str:
        return "observation"


Event = Action | Observation
Subscriber = Callable[[Event], None]


class EventStream:
    def __init__(self, trajectory_path: str | Path | None = None) -> None:
        self._events: list[Event] = []
        self._subscribers: list[Subscriber] = []
        self._path = Path(trajectory_path) if trajectory_path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def subscribe(self, fn: Subscriber) -> None:
        self._subscribers.append(fn)

    def emit(self, event: Event) -> None:
        self._events.append(event)
        if self._path:
            record = {"kind": event.kind, **asdict(event)}
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for fn in self._subscribers:
            fn(event)

    @staticmethod
    def load(trajectory_path: str | Path) -> list[Event]:
        events: list[Event] = []
        for line in Path(trajectory_path).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            kind = record.pop("kind")
            if kind == "action":
                events.append(Action(**record))
            else:
                events.append(Observation(**record))
        return events
