# tests/test_improve.py
from mnemo import improve


def test_analyze_mines_corrections():
    traces = [
        {"input": "log my gym session", "reply": "no tool for that", "ok": False,
         "correction": "store it as a fact: gym leg day done"},
        {"input": "hi", "reply": "hello", "ok": True},
    ]
    examples = improve.analyze(traces)
    assert any("gym" in e for e in examples)
