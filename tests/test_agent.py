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


def test_agent_injects_recalled_facts_into_context(tmp_path):
    """Spec §4: handle() proactively recalls relevant facts and injects them into
    context, so recall does not depend on the small model choosing to call the
    recall tool (the R2 failure observed live with LFM2)."""
    mem = LanceMemory(fake_embed, str(tmp_path / "m"))
    mem.remember("my landlord is Mr Okafor", {})
    captured = {}

    def fake_chat(messages, tools=None):
        captured["messages"] = messages
        return {"content": "Your landlord is Mr Okafor.", "tool_calls": []}

    with patch("mnemo.agent.model.chat", side_effect=fake_chat):
        reply = Agent(mem).handle("who is my landlord?")

    blob = " ".join(
        m["content"] for m in captured["messages"] if isinstance(m.get("content"), str))
    assert "Okafor" in blob  # the recalled fact reached the model's context
    assert "Okafor" in reply


def test_agent_records_assistant_tool_call_in_history(tmp_path):
    """Regression: the assistant's tool-call turn must be appended to the
    conversation before the tool result, or the model re-emits the same call
    every step and never finalizes (observed live with LFM2)."""
    mem = LanceMemory(fake_embed, str(tmp_path / "m"))
    seen_roles = []
    seq = [
        {"content": "", "tool_calls": [{"name": "remember",
            "arguments": {"text": "x"}}]},
        {"content": "done", "tool_calls": []},
    ]
    calls = iter(seq)

    def recorder(messages, tools=None):
        seen_roles.append([m.get("role") for m in messages])
        return next(calls)

    with patch("mnemo.agent.model.chat", side_effect=recorder):
        Agent(mem).handle("remember x")

    # On the second model call, history must contain the assistant tool-call
    # turn AND the tool observation.
    assert "assistant" in seen_roles[1] and "tool" in seen_roles[1]
