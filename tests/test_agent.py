import json

from agentkit.agent import Agent
from agentkit.events import Action, EventStream, Observation
from agentkit.llm import LLMResponse, Message, ToolCall
from agentkit.tools import ToolRegistry, tool


class FakeLLM:
    """Scripted LLM that returns queued responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.received: list[list[Message]] = []

    async def complete(self, messages, tools=None):  # type: ignore[no-untyped-def]
        self.received.append(list(messages))
        return self.responses.pop(0)


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def make_registry() -> ToolRegistry:
    return ToolRegistry([tool(add)])


async def test_agent_calls_tool_then_finishes(tmp_path):
    llm = FakeLLM(
        [
            LLMResponse(
                content="I should add the numbers.",
                tool_calls=[ToolCall(id="1", name="add", args={"a": 2, "b": 3})],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="2", name="finish", args={"answer": "5"})],
            ),
        ]
    )
    stream = EventStream(trajectory_path=tmp_path / "traj.jsonl")
    agent = Agent(llm=llm, tools=make_registry(), event_stream=stream)
    result = await agent.run("What is 2 + 3?")

    assert result.finished
    assert result.answer == "5"
    assert result.steps == 2

    kinds = [e.kind for e in stream.events]
    assert kinds == ["action", "observation", "action"]
    obs = stream.events[1]
    assert isinstance(obs, Observation)
    assert obs.content == "5"

    # tool result was fed back to the LLM
    tool_msgs = [m for m in llm.received[1] if m.role == "tool"]
    assert tool_msgs[0].content == "5"

    # trajectory is replayable
    loaded = EventStream.load(tmp_path / "traj.jsonl")
    assert len(loaded) == 3
    assert isinstance(loaded[0], Action)
    assert loaded[0].tool_name == "add"


async def test_agent_direct_answer_without_tools():
    llm = FakeLLM([LLMResponse(content="Paris")])
    agent = Agent(llm=llm, tools=make_registry())
    result = await agent.run("Capital of France?")
    assert result.finished
    assert result.answer == "Paris"
    assert result.steps == 1


async def test_agent_stops_at_max_steps():
    call = ToolCall(id="x", name="add", args={"a": 1, "b": 1})
    llm = FakeLLM([LLMResponse(content="", tool_calls=[call])] * 3)
    agent = Agent(llm=llm, tools=make_registry(), max_steps=3)
    result = await agent.run("loop forever")
    assert not result.finished
    assert result.steps == 3


async def test_unknown_tool_returns_error_observation():
    reg = make_registry()
    obs = await reg.execute("nope", {})
    assert obs.is_error
    assert "unknown tool" in obs.content


async def test_tool_exception_becomes_error_observation():
    def boom() -> str:
        """Always fails."""
        raise ValueError("bad input")

    reg = ToolRegistry([tool(boom)])
    obs = await reg.execute("boom", {})
    assert obs.is_error
    assert "ValueError: bad input" in obs.content


def test_tool_schema_generation():
    t = tool(add)
    schema = t.schema()
    fn = schema["function"]
    assert fn["name"] == "add"
    assert fn["description"] == "Add two integers."
    params = fn["parameters"]
    assert params["properties"] == {"a": {"type": "integer"}, "b": {"type": "integer"}}
    assert params["required"] == ["a", "b"]
    json.dumps(schema)  # must be JSON-serializable
