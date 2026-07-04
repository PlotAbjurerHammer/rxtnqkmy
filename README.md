# agentkit

A lightweight LLM agent framework in Python. Inspired by OpenHands, smolagents, and LangGraph.

## Features (P0 MVP)

- **ReAct agent loop** — think → act → observe, with step limits (`agentkit/agent.py`)
- **Tool system** — register plain functions as tools; JSON schemas generated from type annotations (`agentkit/tools.py`)
- **LLM abstraction** — one async interface for any OpenAI-compatible API, with retries and token accounting (`agentkit/llm.py`)
- **Event stream** — every Action/Observation is recorded to a replayable JSONL trajectory (`agentkit/events.py`)
- **Built-in tools** — shell, read_file, write_file, fetch_url (`agentkit/builtin_tools.py`)

## Enterprise controls (P1)

- **RunPolicy** (`agentkit/policy.py`) — hard limits: `max_steps`, `max_total_tokens`, `max_wall_clock_seconds`, per-tool `tool_timeout_seconds`. Violations stop the run with a `stop_reason`.
- **Guardrail** (`agentkit/policy.py`) — tool allowlist/denylist and human-in-the-loop approval: tools in `require_approval` are only run if the async `approval_hook(action)` returns True.
- **ContextManager** (`agentkit/context.py`) — clips oversized tool outputs and elides the oldest tool exchanges to keep history within `max_context_tokens` (system prompt, task, and recent messages are always preserved).
- **Checkpointer** (`agentkit/checkpoint.py`) — persists run state (messages, step, token count) as JSON; interrupted runs resume via `agent.run(task, resume=True)`.

```python
from agentkit import Agent, Checkpointer, Guardrail, RunPolicy

agent = Agent(
    llm=llm,
    tools=registry,
    policy=RunPolicy(max_total_tokens=200_000, max_wall_clock_seconds=600),
    guardrail=Guardrail(denied_tools={"shell"}, require_approval={"write_file"},
                        approval_hook=my_async_approver),
    checkpointer=Checkpointer("run_state.json"),
)
result = await agent.run(task, resume=True)
print(result.stop_reason, result.total_tokens)
```

CLI flags: `--max-total-tokens`, `--timeout`, `--tool-timeout`, `--deny-tool NAME`, `--checkpoint PATH`, `--resume`.

Try it without an API key using the scripted mock server:

```bash
python examples/mock_server.py &
AGENTKIT_API_KEY=demo agentkit run "列出目录文件并写一个摘要" --base-url http://127.0.0.1:8901
```

## Install

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
export OPENAI_API_KEY=sk-...
agentkit run "List the files in the current directory and summarize them" \
  --model gpt-4o-mini --trajectory traj.jsonl
```

### Library

```python
import asyncio
from agentkit import Agent, OpenAICompatibleLLM, ToolRegistry, default_tools, tool

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

llm = OpenAICompatibleLLM(model="gpt-4o-mini", api_key="sk-...")
registry = ToolRegistry([*default_tools(), tool(add)])
agent = Agent(llm=llm, tools=registry, max_steps=20)

result = asyncio.run(agent.run("What is 21 + 21?"))
print(result.answer)
```

## Development

```bash
ruff check .
mypy agentkit
pytest -q
```

## Roadmap

- P1 (remaining): MCP client support, Docker sandbox, memory (mem0-style)
- P2: graph workflow orchestration (LangGraph-style), multi-agent delegation
- P3: eval harness (GAIA/SWE-bench subsets), web UI, OpenTelemetry tracing
