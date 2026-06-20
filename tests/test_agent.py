# tests/test_agent.py
from unittest.mock import patch

from mnemo.agent import Agent
from mnemo.memory import LanceMemory


def fake_embed(t):
    return [float(t.count(c)) for c in "abcdefghij"]


def test_agent_remembers_then_recalls(tmp_path):
    mem = LanceMemory(fake_embed, str(tmp_path / "m"))
    seq = [
        {"content": "", "tool_calls": [{"name": "remember",
            "arguments": {"text": "my cat is named Pixel"}}]},
        {"content": "Got it, I'll remember Pixel.", "tool_calls": []},
    ]
    with patch("mnemo.agent.model.chat", side_effect=seq):
        reply = Agent(mem).handle("remember my cat is named Pixel")
    assert "Pixel" in reply
    assert any("Pixel" in f.text for f in mem.all_facts())
