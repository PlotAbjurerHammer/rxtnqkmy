"""Run policies and guardrails for enterprise deployments.

RunPolicy enforces hard resource limits on an agent run (steps, tokens,
wall-clock time, per-tool timeout). Guardrail controls which tools may run
and supports a human-in-the-loop approval hook for sensitive tools.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agentkit.events import Action


class PolicyViolation(Exception):
    """Raised when a run exceeds a hard resource limit."""


@dataclass
class RunPolicy:
    max_steps: int = 30
    max_total_tokens: int | None = None
    max_wall_clock_seconds: float | None = None
    tool_timeout_seconds: float | None = 60.0

    def start(self) -> _PolicyState:
        return _PolicyState(policy=self, started_at=time.monotonic())


@dataclass
class _PolicyState:
    policy: RunPolicy
    started_at: float
    total_tokens: int = 0

    def record_tokens(self, prompt: int, completion: int) -> None:
        self.total_tokens += prompt + completion

    def check(self) -> None:
        p = self.policy
        if p.max_total_tokens is not None and self.total_tokens > p.max_total_tokens:
            raise PolicyViolation(
                f"token budget exceeded: {self.total_tokens} > {p.max_total_tokens}"
            )
        if p.max_wall_clock_seconds is not None:
            elapsed = time.monotonic() - self.started_at
            if elapsed > p.max_wall_clock_seconds:
                raise PolicyViolation(
                    f"wall-clock limit exceeded: {elapsed:.1f}s > {p.max_wall_clock_seconds}s"
                )


ApprovalHook = Callable[[Action], Awaitable[bool]]


@dataclass
class Guardrail:
    """Controls which tools the agent may execute.

    - `allowed_tools`: if set, only these tools may run (allowlist).
    - `denied_tools`: these tools are always rejected (denylist).
    - `require_approval`: tools that must be approved by `approval_hook`
      before running (human-in-the-loop).
    """

    allowed_tools: set[str] | None = None
    denied_tools: set[str] = field(default_factory=set)
    require_approval: set[str] = field(default_factory=set)
    approval_hook: ApprovalHook | None = None

    async def authorize(self, action: Action) -> str | None:
        """Return None if the action may run, or a rejection reason string."""
        name = action.tool_name
        if name in self.denied_tools:
            return f"tool '{name}' is denied by guardrail policy"
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return f"tool '{name}' is not in the allowed tool list"
        if name in self.require_approval:
            if self.approval_hook is None:
                return f"tool '{name}' requires approval but no approval hook is configured"
            approved = await self.approval_hook(action)
            if not approved:
                return f"tool '{name}' was rejected by the approver"
        return None
