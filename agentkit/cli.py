"""Command-line entry point: `agentkit run "your task"`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agentkit.agent import Agent
from agentkit.builtin_tools import default_tools
from agentkit.events import Action, EventStream, Observation
from agentkit.llm import OpenAICompatibleLLM
from agentkit.tools import ToolRegistry


def _print_event(event: Action | Observation) -> None:
    if isinstance(event, Action):
        print(f"[action] {event.tool_name}({event.args})")
        if event.thought:
            print(f"[thought] {event.thought}")
    else:
        prefix = "[error]" if event.is_error else "[observation]"
        print(f"{prefix} {event.content[:500]}")


async def _run(task: str, model: str, base_url: str, max_steps: int, trajectory: str) -> int:
    api_key = os.environ.get("AGENTKIT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("error: set AGENTKIT_API_KEY or OPENAI_API_KEY", file=sys.stderr)
        return 1
    llm = OpenAICompatibleLLM(model=model, api_key=api_key, base_url=base_url)
    events = EventStream(trajectory_path=trajectory)
    events.subscribe(_print_event)
    agent = Agent(
        llm=llm,
        tools=ToolRegistry(default_tools()),
        max_steps=max_steps,
        event_stream=events,
    )
    result = await agent.run(task)
    print(f"\n=== answer (steps: {result.steps}, finished: {result.finished}) ===")
    print(result.answer)
    return 0 if result.finished else 2


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentkit")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the agent on a task")
    run.add_argument("task")
    run.add_argument("--model", default="gpt-4o-mini")
    run.add_argument("--base-url", default="https://api.openai.com/v1")
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--trajectory", default="trajectory.jsonl")
    args = parser.parse_args()
    sys.exit(
        asyncio.run(_run(args.task, args.model, args.base_url, args.max_steps, args.trajectory))
    )


if __name__ == "__main__":
    main()
