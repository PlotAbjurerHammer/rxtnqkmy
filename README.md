# agentkit

A lightweight LLM agent framework in Python. Inspired by OpenHands, smolagents, and LangGraph.

## Features (P0 MVP)

- **ReAct agent loop** — think → act → observe, with step limits (`agentkit/agent.py`)
- **Tool system** — register plain functions as tools; JSON schemas generated from type annotations (`agentkit/tools.py`)
- **LLM abstraction** — one async interface for any OpenAI-compatible API, with retries and token accounting (`agentkit/llm.py`)
- **Event stream** — every Action/Observation is recorded to a replayable JSONL trajectory (`agentkit/events.py`)
- **Built-in tools** — shell, read_file, write_file, fetch_url (`agentkit/builtin_tools.py`)

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

- P1: MCP client support, Docker sandbox, context compression, memory (mem0-style)
- P2: graph workflow orchestration (LangGraph-style), multi-agent delegation, checkpoints
- P3: eval harness (GAIA/SWE-bench subsets), web UI, OpenTelemetry tracing
