"""Local mock OpenAI-compatible server for demoing agentkit without an API key.

Scripts a 3-step run: shell -> write_file -> finish.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

STEPS = [
    {
        "content": "先看看当前目录里有什么文件。",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": "ls"})},
            }
        ],
    },
    {
        "content": "把文件清单写入 summary.txt。",
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps(
                        {
                            "path": "demo_output/summary.txt",
                            "content": "agentkit demo: listed repo files",
                        }
                    ),
                },
            }
        ],
    },
    {
        "content": "",
        "tool_calls": [
            {
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": json.dumps(
                        {"answer": "已列出目录文件并将摘要写入 demo_output/summary.txt，任务完成。"}
                    ),
                },
            }
        ],
    },
]

step = 0


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global step
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        message = STEPS[min(step, len(STEPS) - 1)]
        step += 1
        body = json.dumps(
            {
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8901), Handler).serve_forever()
