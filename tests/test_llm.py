from agentkit.llm import Message, OpenAICompatibleLLM


def test_message_to_dict_includes_tool_fields():
    m = Message(role="tool", content="ok", tool_call_id="abc", name="shell")
    assert m.to_dict() == {
        "role": "tool",
        "content": "ok",
        "tool_call_id": "abc",
        "name": "shell",
    }


def test_parse_response_with_tool_calls():
    llm = OpenAICompatibleLLM(model="m", api_key="k")
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "shell", "arguments": '{"command": "ls"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    resp = llm._parse(data)
    assert resp.content == ""
    assert resp.tool_calls[0].name == "shell"
    assert resp.tool_calls[0].args == {"command": "ls"}
    assert llm.total_prompt_tokens == 10
    assert llm.total_completion_tokens == 5
