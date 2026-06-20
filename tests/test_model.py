from unittest.mock import patch
from mnemo import model


def test_chat_parses_tool_calls():
    fake = {"message": {"content": "", "tool_calls": [
        {"function": {"name": "remember", "arguments": {"text": "x"}}}]}}
    with patch("mnemo.model.ollama.chat", return_value=fake):
        out = model.chat([{"role": "user", "content": "remember x"}], tools=[{"x": 1}])
    assert out["tool_calls"][0]["name"] == "remember"
    assert out["tool_calls"][0]["arguments"] == {"text": "x"}


def test_chat_parses_stringified_arguments():
    fake = {"message": {"content": "", "tool_calls": [
        {"function": {"name": "recall", "arguments": '{"query": "cat"}'}}]}}
    with patch("mnemo.model.ollama.chat", return_value=fake):
        out = model.chat([{"role": "user", "content": "find cat"}])
    assert out["tool_calls"][0]["arguments"] == {"query": "cat"}


def test_embed_returns_vector():
    with patch("mnemo.model.ollama.embeddings", return_value={"embedding": [0.1, 0.2]}):
        v = model.embed("hello")
    assert v == [0.1, 0.2]
