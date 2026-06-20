# tests/test_tools.py
# Real LanceMemory (real embeddings injected), no mocks.
from mnemo import tools
from mnemo.memory import LanceMemory


def fake_embed(t):
    return [float(t.count(c)) for c in "abcdefghij"]


def test_remember_dedupes_identical_fact(tmp_path):
    mem = LanceMemory(fake_embed, str(tmp_path / "m"))
    r1 = tools.run("remember", {"text": "my car is a blue Honda"}, mem)
    r2 = tools.run("remember", {"text": "my car is a blue Honda"}, mem)
    assert r1.startswith("stored:")
    assert r2.startswith("already known:")
    # only one copy persisted
    assert sum("blue Honda" in f.text for f in mem.all_facts()) == 1


def test_capture_photo_tool_registered():
    names = {s["function"]["name"] for s in tools.SCHEMAS}
    assert "capture_photo" in names and "ingest_photo" in names
