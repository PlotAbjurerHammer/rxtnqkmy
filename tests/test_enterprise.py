import asyncio

from agentkit.agent import Agent
from agentkit.checkpoint import Checkpointer
from agentkit.context import ELIDED_NOTICE, TRUNCATION_NOTICE, ContextManager
from agentkit.llm import LLMResponse, Message, ToolCall
from agentkit.policy import Guardrail, RunPolicy
from agentkit.tools import ToolRegistry, tool
from tests.test_agent import FakeLLM, make_registry


def add_call(call_id: str = "x") -> ToolCall:
    return ToolCall(id=call_id, name="add", args={"a": 1, "b": 1})


def finish_call(answer: str) -> ToolCall:
    return ToolCall(id="f", name="finish", args={"answer": answer})


def burn_response(tokens: int) -> LLMResponse:
    return LLMResponse(content="", tool_calls=[add_call()], prompt_tokens=tokens)


async def test_token_budget_stops_run():
    llm = FakeLLM([burn_response(600), burn_response(600)])
    agent = Agent(llm=llm, tools=make_registry(), policy=RunPolicy(max_total_tokens=1000))
    result = await agent.run("burn tokens")
    assert not result.finished
    assert "token budget exceeded" in result.stop_reason
    assert result.total_tokens == 1200


async def test_wall_clock_limit_stops_run():
    llm = FakeLLM([LLMResponse(content="", tool_calls=[add_call()])] * 2)
    agent = Agent(
        llm=llm, tools=make_registry(), policy=RunPolicy(max_wall_clock_seconds=-1.0)
    )
    result = await agent.run("too slow")
    assert not result.finished
    assert "wall-clock limit exceeded" in result.stop_reason


async def test_tool_timeout_becomes_error_observation():
    async def slow() -> str:
        """Sleeps forever."""
        await asyncio.sleep(10)
        return "done"

    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=[ToolCall(id="1", name="slow", args={})]),
            LLMResponse(content="", tool_calls=[finish_call("ok")]),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=ToolRegistry([tool(slow)]),
        policy=RunPolicy(tool_timeout_seconds=0.01),
    )
    result = await agent.run("run slow tool")
    assert result.finished
    tool_msgs = [m for m in llm.received[1] if m.role == "tool"]
    assert "timed out" in tool_msgs[0].content


async def test_guardrail_denylist_blocks_tool():
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=[add_call()]),
            LLMResponse(content="", tool_calls=[finish_call("ok")]),
        ]
    )
    agent = Agent(llm=llm, tools=make_registry(), guardrail=Guardrail(denied_tools={"add"}))
    result = await agent.run("try denied tool")
    assert result.finished
    tool_msgs = [m for m in llm.received[1] if m.role == "tool"]
    assert "denied by guardrail" in tool_msgs[0].content


async def test_guardrail_approval_hook():
    approvals: list[str] = []

    async def approve(action):  # type: ignore[no-untyped-def]
        approvals.append(action.tool_name)
        return action.tool_name == "add"

    guardrail = Guardrail(require_approval={"add"}, approval_hook=approve)
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=[add_call()]),
            LLMResponse(content="", tool_calls=[finish_call("2")]),
        ]
    )
    agent = Agent(llm=llm, tools=make_registry(), guardrail=guardrail)
    result = await agent.run("needs approval")
    assert result.finished
    assert approvals == ["add"]
    tool_msgs = [m for m in llm.received[1] if m.role == "tool"]
    assert tool_msgs[0].content == "2"


async def test_guardrail_allowlist_rejects_unknown():
    guardrail = Guardrail(allowed_tools={"read_file"})
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=[add_call()]),
            LLMResponse(content="", tool_calls=[finish_call("ok")]),
        ]
    )
    agent = Agent(llm=llm, tools=make_registry(), guardrail=guardrail)
    result = await agent.run("try disallowed tool")
    assert result.finished
    tool_msgs = [m for m in llm.received[1] if m.role == "tool"]
    assert "not in the allowed tool list" in tool_msgs[0].content


def test_context_clip_observation():
    cm = ContextManager(max_observation_chars=10)
    clipped = cm.clip_observation("x" * 100)
    assert clipped.startswith("x" * 10)
    assert clipped.endswith(TRUNCATION_NOTICE)


def test_context_fit_elides_old_tool_output():
    cm = ContextManager(max_context_tokens=100, keep_recent_messages=1)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content="calling tool", tool_calls=[]),
        Message(role="tool", content="y" * 2000, tool_call_id="1", name="add"),
        Message(role="assistant", content="done"),
    ]
    fitted = cm.fit(messages)
    assert fitted[3].content == ELIDED_NOTICE
    assert fitted[0].content == "sys"
    assert fitted[4].content == "done"


async def test_checkpoint_resume(tmp_path):
    ckpt = Checkpointer(tmp_path / "state.json")

    # First run: exhausts max_steps and saves a checkpoint.
    llm1 = FakeLLM([LLMResponse(content="", tool_calls=[add_call()])])
    agent1 = Agent(llm=llm1, tools=make_registry(), checkpointer=ckpt, max_steps=1)
    result1 = await agent1.run("resumable task")
    assert not result1.finished
    assert ckpt.load() is not None

    # Second run resumes: history includes the previous tool exchange.
    llm2 = FakeLLM([LLMResponse(content="", tool_calls=[finish_call("2")])])
    agent2 = Agent(llm=llm2, tools=make_registry(), checkpointer=ckpt, max_steps=5)
    result2 = await agent2.run("resumable task", resume=True)
    assert result2.finished
    assert result2.answer == "2"
    assert result2.steps == 2
    assert any(m.role == "tool" for m in llm2.received[0])
    # Checkpoint cleared after successful finish.
    assert ckpt.load() is None
