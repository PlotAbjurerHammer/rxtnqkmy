"""Desktop GUI (Tkinter): run agent tasks with a live event log and
human-in-the-loop approval dialogs for sensitive tools.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from agentkit.agent import Agent, AgentResult
from agentkit.builtin_tools import default_tools
from agentkit.events import Action, EventStream, Observation
from agentkit.llm import OpenAICompatibleLLM
from agentkit.policy import Guardrail, RunPolicy
from agentkit.tools import ToolRegistry

APPROVAL_TOOLS = {"shell", "write_file"}


class AgentGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("agentkit — Desktop Agent")
        root.geometry("900x640")

        self._event_queue: queue.Queue[Action | Observation | AgentResult] = queue.Queue()
        self._approval_request: queue.Queue[Action] = queue.Queue()
        self._approval_response: queue.Queue[bool] = queue.Queue()
        self._worker: threading.Thread | None = None

        config = ttk.Frame(root, padding=8)
        config.pack(fill=tk.X)
        ttk.Label(config, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value="gpt-4o-mini")
        ttk.Entry(config, textvariable=self.model_var, width=24).grid(row=0, column=1, padx=4)
        ttk.Label(config, text="Base URL:").grid(row=0, column=2, sticky=tk.W)
        self.base_url_var = tk.StringVar(value="https://api.openai.com/v1")
        ttk.Entry(config, textvariable=self.base_url_var, width=36).grid(row=0, column=3, padx=4)
        self.approval_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config, text="敏感工具需审批 (shell/write_file)", variable=self.approval_var
        ).grid(row=0, column=4, padx=8)

        task_frame = ttk.Frame(root, padding=8)
        task_frame.pack(fill=tk.X)
        ttk.Label(task_frame, text="任务:").pack(side=tk.LEFT)
        self.task_var = tk.StringVar()
        entry = ttk.Entry(task_frame, textvariable=self.task_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        entry.bind("<Return>", lambda _e: self.start())
        self.run_button = ttk.Button(task_frame, text="运行", command=self.start)
        self.run_button.pack(side=tk.LEFT)

        self.log = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.log.tag_configure("action", foreground="#0057b7")
        self.log.tag_configure("thought", foreground="#666666")
        self.log.tag_configure("observation", foreground="#1a7f37")
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("result", foreground="#6a1b9a", font=("TkDefaultFont", 10, "bold"))

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(root, textvariable=self.status_var, padding=4).pack(fill=tk.X)

        root.after(100, self._poll)

    def start(self) -> None:
        task = self.task_var.get().strip()
        if not task or (self._worker and self._worker.is_alive()):
            return
        api_key = os.environ.get("AGENTKIT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            messagebox.showerror("agentkit", "请设置 AGENTKIT_API_KEY 或 OPENAI_API_KEY 环境变量")
            return
        self.run_button.state(["disabled"])
        self.status_var.set(f"运行中: {task}")
        self._append(f"\n=== 任务: {task} ===\n", "result")
        self._worker = threading.Thread(
            target=self._run_agent, args=(task, api_key), daemon=True
        )
        self._worker.start()

    def _run_agent(self, task: str, api_key: str) -> None:
        llm = OpenAICompatibleLLM(
            model=self.model_var.get(), api_key=api_key, base_url=self.base_url_var.get()
        )
        events = EventStream()
        events.subscribe(self._event_queue.put)
        guardrail = None
        if self.approval_var.get():
            guardrail = Guardrail(
                require_approval=APPROVAL_TOOLS, approval_hook=self._request_approval
            )
        agent = Agent(
            llm=llm,
            tools=ToolRegistry(default_tools()),
            policy=RunPolicy(max_steps=30),
            guardrail=guardrail,
            event_stream=events,
        )
        try:
            result = asyncio.run(agent.run(task))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            result = AgentResult(answer=f"运行失败: {exc}", steps=0, finished=False,
                                 stop_reason="exception")
        self._event_queue.put(result)

    async def _request_approval(self, action: Action) -> bool:
        self._approval_request.put(action)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._approval_response.get)

    def _poll(self) -> None:
        try:
            while True:
                action = self._approval_request.get_nowait()
                approved = messagebox.askyesno(
                    "工具审批",
                    f"Agent 请求执行敏感工具:\n\n{action.tool_name}({action.args})\n\n是否允许？",
                )
                self._approval_response.put(approved)
        except queue.Empty:
            pass
        try:
            while True:
                item = self._event_queue.get_nowait()
                self._render(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _render(self, item: Action | Observation | AgentResult) -> None:
        if isinstance(item, Action):
            if item.thought:
                self._append(f"[思考] {item.thought}\n", "thought")
            self._append(f"[动作] {item.tool_name}({item.args})\n", "action")
        elif isinstance(item, Observation):
            tag = "error" if item.is_error else "observation"
            label = "错误" if item.is_error else "结果"
            self._append(f"[{label}] {item.content[:1000]}\n", tag)
        else:
            self._append(
                f"\n=== 完成 (步数: {item.steps}, tokens: {item.total_tokens}, "
                f"stop: {item.stop_reason}) ===\n{item.answer}\n",
                "result",
            )
            self.run_button.state(["!disabled"])
            self.status_var.set("就绪")

    def _append(self, text: str, tag: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text, tag)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    AgentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
