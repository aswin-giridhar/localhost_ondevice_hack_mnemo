# tests/test_offline_proof.py
import socket

from mnemo.memory import LanceMemory


def test_recall_works_with_network_blocked(tmp_path, monkeypatch):
    def no_net(*a, **k):
        raise OSError("network disabled for offline proof")

    monkeypatch.setattr(socket, "create_connection", no_net)

    def fe(t):
        return [float(t.count(c)) for c in "abcdefghij"]

    m = LanceMemory(fe, str(tmp_path / "m"))
    m.remember("the safe code is 7741", {})
    assert "7741" in m.recall("safe code", 1)[0].text
