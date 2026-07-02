"""Built-in tools: shell execution, file read/write, and web fetching."""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx

from agentkit.tools import Tool, tool


def _run_shell(command: str, timeout: int = 60) -> str:
    """Run a shell command and return its combined stdout/stderr output."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = result.stdout + result.stderr
    return f"exit code: {result.returncode}\n{output.strip()}"


def _read_file(path: str) -> str:
    """Read a text file and return its contents."""
    return Path(path).read_text(encoding="utf-8")


def _write_file(path: str, content: str) -> str:
    """Write content to a text file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


async def _fetch_url(url: str) -> str:
    """Fetch a URL and return the response body as text (truncated to 20000 chars)."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text[:20000]


shell = tool(_run_shell, name="shell")
read_file = tool(_read_file, name="read_file")
write_file = tool(_write_file, name="write_file")
fetch_url = tool(_fetch_url, name="fetch_url")


def default_tools() -> list[Tool]:
    return [shell, read_file, write_file, fetch_url]
