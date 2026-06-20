# tests/test_selftool.py
import pytest

from mnemo import improve, tools
from mnemo.memory import LanceMemory


def fe(t):
    return [float(t.count(c)) for c in "abcdefghij"]


SAFE = 'def shout(args, mem):\n    return args["text"].upper()\n'
UNSAFE_IMPORT = 'def bad(args, mem):\n    import os\n    return os.listdir(".")\n'
UNSAFE_EVAL = 'def bad(args, mem):\n    return eval(args["text"])\n'
UNSAFE_DUNDER = 'def bad(args, mem):\n    return args.__class__.__bases__\n'


@pytest.fixture(autouse=True)
def _clear_learned():
    tools.LEARNED.clear()
    yield
    tools.LEARNED.clear()


def test_validate_rejects_import():
    with pytest.raises(ValueError):
        improve.validate_tool_code(UNSAFE_IMPORT, "bad")


def test_validate_rejects_eval():
    with pytest.raises(ValueError):
        improve.validate_tool_code(UNSAFE_EVAL, "bad")


def test_validate_rejects_dunder():
    with pytest.raises(ValueError):
        improve.validate_tool_code(UNSAFE_DUNDER, "bad")


def test_validate_rejects_wrong_name():
    with pytest.raises(ValueError):
        improve.validate_tool_code(SAFE, "not_shout")


def test_load_and_call_safe_tool():
    fn = improve.load_tool(SAFE, "shout")
    assert fn({"text": "hi"}, None) == "HI"


def test_register_and_run_learned_tool(tmp_path):
    mem = LanceMemory(fe, str(tmp_path / "m"))
    tools.register("shout", lambda args, mem: args["text"].upper())
    assert tools.run("shout", {"text": "hi"}, mem) == "HI"


def test_propose_and_register_end_to_end(tmp_path):
    def fake_chat(messages, tools=None):
        return {"content": "```python\n" + SAFE + "```", "tool_calls": []}

    path = tmp_path / "learned_tools.py"
    code = improve.propose_and_register(
        "shout", "uppercase the text arg", path=str(path), chat_fn=fake_chat)
    assert "def shout" in code
    assert path.exists() and "def shout" in path.read_text()
    assert tools.run("shout", {"text": "yo"}, None) == "YO"


def test_propose_rejects_unsafe_generated_code(tmp_path):
    def evil_chat(messages, tools=None):
        return {"content": UNSAFE_IMPORT, "tool_calls": []}

    path = tmp_path / "learned_tools.py"
    with pytest.raises(ValueError):
        improve.propose_and_register(
            "bad", "do something", path=str(path), chat_fn=evil_chat)
    # unsafe code must NOT be persisted or registered
    assert "bad" not in tools.LEARNED
    assert not path.exists()
