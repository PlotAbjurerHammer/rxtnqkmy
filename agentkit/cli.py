"""Command-line entry point: `agentkit run "your task"`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agentkit.agent import Agent
from agentkit.builtin_tools import default_tools
from agentkit.checkpoint import Checkpointer
from agentkit.events import Action, EventStream, Observation
from agentkit.llm import OpenAICompatibleLLM
from agentkit.policy import Guardrail, RunPolicy
from agentkit.tools import ToolRegistry


def _print_event(event: Action | Observation) -> None:
    if isinstance(event, Action):
        print(f"[action] {event.tool_name}({event.args})")
        if event.thought:
            print(f"[thought] {event.thought}")
    else:
        prefix = "[error]" if event.is_error else "[observation]"
        print(f"{prefix} {event.content[:500]}")


async def _run(args: argparse.Namespace) -> int:
    api_key = os.environ.get("AGENTKIT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("error: set AGENTKIT_API_KEY or OPENAI_API_KEY", file=sys.stderr)
        return 1
    llm = OpenAICompatibleLLM(model=args.model, api_key=api_key, base_url=args.base_url)
    events = EventStream(trajectory_path=args.trajectory)
    events.subscribe(_print_event)
    policy = RunPolicy(
        max_steps=args.max_steps,
        max_total_tokens=args.max_total_tokens,
        max_wall_clock_seconds=args.timeout,
        tool_timeout_seconds=args.tool_timeout,
    )
    guardrail = Guardrail(denied_tools=set(args.deny_tool)) if args.deny_tool else None
    checkpointer = Checkpointer(args.checkpoint) if args.checkpoint else None
    agent = Agent(
        llm=llm,
        tools=ToolRegistry(default_tools()),
        policy=policy,
        guardrail=guardrail,
        checkpointer=checkpointer,
        event_stream=events,
    )
    result = await agent.run(args.task, resume=args.resume)
    print(
        f"\n=== answer (steps: {result.steps}, tokens: {result.total_tokens}, "
        f"stop: {result.stop_reason}) ==="
    )
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
    run.add_argument("--max-total-tokens", type=int, default=None)
    run.add_argument("--timeout", type=float, default=None, help="wall-clock limit in seconds")
    run.add_argument("--tool-timeout", type=float, default=60.0)
    run.add_argument("--deny-tool", action="append", default=[])
    run.add_argument("--checkpoint", default=None, help="path to checkpoint file")
    run.add_argument("--resume", action="store_true", help="resume from checkpoint")
    run.add_argument("--trajectory", default="trajectory.jsonl")
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
