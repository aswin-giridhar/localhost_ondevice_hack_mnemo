# tests/test_server.py
from unittest.mock import patch

from fastapi.testclient import TestClient


def test_chat_route(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    from mnemo import server
    with patch.object(server.AGENT, "handle", return_value="hello back"):
        c = TestClient(server.app)
        r = c.post("/chat", json={"message": "hi"})
    assert r.status_code == 200 and r.json()["reply"] == "hello back"


def test_health_reports_offline_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    from mnemo import server
    c = TestClient(server.app)
    assert "offline" in c.get("/health").json()


def test_selftool_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    from mnemo import server
    # default flag is off -> the route refuses to author/run code
    server.SETTINGS.enable_selftool = False
    c = TestClient(server.app)
    r = c.post("/selftool", json={"name": "x", "description": "y"})
    assert r.status_code == 200 and r.json()["enabled"] is False
