"""Agent core: a ReAct-style loop that alternates between LLM decisions and
tool executions until the model calls `finish` or a resource limit is reached.

Enterprise controls:
- RunPolicy: hard limits on steps, total tokens, wall-clock time, tool timeout.
- Guardrail: tool allow/deny lists and human-in-the-loop approval.
- ContextManager: keeps history within a token budget.
- Checkpointer: persists run state so interrupted runs can resume.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from agentkit.checkpoint import Checkpointer, RunState
from agentkit.context import ContextManager
from agentkit.events import Action, EventStream, Observation
from agentkit.llm import LLM, Message, ToolCall
from agentkit.policy import Guardrail, PolicyViolation, RunPolicy
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
    stop_reason: str = "finished"
    total_tokens: int = 0
    events: list[Action | Observation] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        policy: RunPolicy | None = None,
        guardrail: Guardrail | None = None,
        context: ContextManager | None = None,
        checkpointer: Checkpointer | None = None,
        event_stream: EventStream | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.policy = policy or RunPolicy()
        if max_steps is not None:
            self.policy.max_steps = max_steps
        self.guardrail = guardrail
        self.context = context or ContextManager()
        self.checkpointer = checkpointer
        self.events = event_stream or EventStream()

    async def run(self, task: str, resume: bool = False) -> AgentResult:
        state = self._initial_state(task, resume)
        policy_state = self.policy.start()
        policy_state.total_tokens = state.total_tokens
        tool_schemas = [*self.tools.schemas(), FINISH_TOOL_SCHEMA]

        while state.step < self.policy.max_steps:
            state.step += 1
            state.messages = self.context.fit(state.messages)

            response = await self.llm.complete(state.messages, tools=tool_schemas)
            policy_state.record_tokens(response.prompt_tokens, response.completion_tokens)
            state.total_tokens = policy_state.total_tokens
            try:
                policy_state.check()
            except PolicyViolation as violation:
                return self._stop(state, str(violation))

            if not response.tool_calls:
                # Model answered directly without calling finish.
                self.events.emit(Action(tool_name="finish", args={}, thought=response.content))
                return self._finish(state, response.content)

            state.messages.append(self._assistant_message(response.content, response.tool_calls))
            for call in response.tool_calls:
                action = Action(tool_name=call.name, args=call.args, thought=response.content)
                self.events.emit(action)

                if call.name == "finish":
                    return self._finish(state, str(call.args.get("answer", "")))

                observation = await self._execute(action, call)
                self.events.emit(observation)
                state.messages.append(
                    Message(
                        role="tool",
                        content=self.context.clip_observation(observation.content),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

            if self.checkpointer:
                self.checkpointer.save(state)

        return self._stop(state, f"max steps reached ({self.policy.max_steps})")

    async def _execute(self, action: Action, call: ToolCall) -> Observation:
        if self.guardrail:
            rejection = await self.guardrail.authorize(action)
            if rejection:
                return Observation(content=rejection, is_error=True)
        timeout = self.policy.tool_timeout_seconds
        try:
            return await asyncio.wait_for(
                self.tools.execute(call.name, call.args), timeout=timeout
            )
        except TimeoutError:
            return Observation(
                content=f"tool '{call.name}' timed out after {timeout}s", is_error=True
            )

    def _initial_state(self, task: str, resume: bool) -> RunState:
        if resume and self.checkpointer:
            saved = self.checkpointer.load()
            if saved is not None and saved.task == task:
                return saved
        return RunState(
            task=task,
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=task),
            ],
        )

    def _finish(self, state: RunState, answer: str) -> AgentResult:
        if self.checkpointer:
            self.checkpointer.clear()
        return AgentResult(
            answer=answer,
            steps=state.step,
            finished=True,
            stop_reason="finished",
            total_tokens=state.total_tokens,
            events=self.events.events,
        )

    def _stop(self, state: RunState, reason: str) -> AgentResult:
        if self.checkpointer:
            self.checkpointer.save(state)
        return AgentResult(
            answer=reason,
            steps=state.step,
            finished=False,
            stop_reason=reason,
            total_tokens=state.total_tokens,
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
