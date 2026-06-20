# tests/test_trace.py
from mnemo.trace import TraceLog


def test_log_and_read(tmp_path):
    t = TraceLog(str(tmp_path / "traces.jsonl"))
    t.log({"input": "hi", "ok": True})
    t.log({"input": "bye", "ok": False})
    rows = TraceLog(str(tmp_path / "traces.jsonl")).turns()
    assert len(rows) == 2 and rows[0]["seq"] == 0 and rows[1]["ok"] is False
