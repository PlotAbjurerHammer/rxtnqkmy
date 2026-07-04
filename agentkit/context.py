"""Context management: keep the conversation within a token budget by
truncating oversized tool outputs and dropping the oldest completed
tool exchanges, while always preserving the system prompt and the task.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentkit.llm import Message

TRUNCATION_NOTICE = "\n...[truncated by context manager]"
ELIDED_NOTICE = "[earlier tool output elided to fit the context budget]"


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


@dataclass
class ContextManager:
    max_context_tokens: int = 64_000
    max_observation_chars: int = 8_000
    keep_recent_messages: int = 6

    def clip_observation(self, content: str) -> str:
        if len(content) <= self.max_observation_chars:
            return content
        return content[: self.max_observation_chars] + TRUNCATION_NOTICE

    def fit(self, messages: list[Message]) -> list[Message]:
        """Elide the oldest tool outputs until the history fits the budget.

        The system prompt (index 0), the user task (index 1), and the most
        recent `keep_recent_messages` messages are never elided.
        """
        if self._total_tokens(messages) <= self.max_context_tokens:
            return messages
        result = list(messages)
        protected_tail = max(self.keep_recent_messages, 0)
        for i in range(2, len(result) - protected_tail):
            msg = result[i]
            if msg.role == "tool" and msg.content != ELIDED_NOTICE:
                result[i] = Message(
                    role="tool",
                    content=ELIDED_NOTICE,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
                if self._total_tokens(result) <= self.max_context_tokens:
                    break
        return result

    @staticmethod
    def _total_tokens(messages: list[Message]) -> int:
        return sum(estimate_tokens(m.content) for m in messages)
