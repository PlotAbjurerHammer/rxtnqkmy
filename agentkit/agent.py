"""Agent core: a ReAct-style loop that alternates between LLM decisions and
tool executions until the model calls `finish` or a step limit is reached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agentkit.events import Action, EventStream, Observation
from agentkit.llm import LLM, Message, ToolCall
from agentkit.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """\
You are a capable autonomous agent. Solve the user's task step by step.
Use the provided tools to gather information and take actions.
When the task is complete, call the `finish` tool with your final answer.
"""

FINISH_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call when the task is complete, with the final answer.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
}


@dataclass
class AgentResult:
    answer: str
    steps: int
    finished: bool
    events: list[Action | Observation] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 30,
        event_stream: EventStream | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.events = event_stream or EventStream()

    async def run(self, task: str) -> AgentResult:
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task),
        ]
        tool_schemas = [*self.tools.schemas(), FINISH_TOOL_SCHEMA]

        for step in range(1, self.max_steps + 1):
            response = await self.llm.complete(messages, tools=tool_schemas)

            if not response.tool_calls:
                # Model answered directly without calling finish.
                self.events.emit(Action(tool_name="finish", args={}, thought=response.content))
                return AgentResult(
                    answer=response.content, steps=step, finished=True, events=self.events.events
                )

            messages.append(self._assistant_message(response.content, response.tool_calls))
            for call in response.tool_calls:
                action = Action(tool_name=call.name, args=call.args, thought=response.content)
                self.events.emit(action)

                if call.name == "finish":
                    answer = str(call.args.get("answer", ""))
                    return AgentResult(
                        answer=answer, steps=step, finished=True, events=self.events.events
                    )

                observation = await self.tools.execute(call.name, call.args)
                self.events.emit(observation)
                messages.append(
                    Message(
                        role="tool",
                        content=observation.content,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        return AgentResult(
            answer="max steps reached", steps=self.max_steps, finished=False,
            events=self.events.events,
        )

    @staticmethod
    def _assistant_message(content: str, tool_calls: list[ToolCall]) -> Message:
        calls: list[dict[str, object]] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
            }
            for tc in tool_calls
        ]
        return Message(role="assistant", content=content, tool_calls=calls)
