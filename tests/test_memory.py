# tests/test_memory.py
from mnemo.memory import LanceMemory


def fake_embed(text: str):
    return [float(text.count(c)) for c in "abcdefghij"]


def test_remember_recall_roundtrip(tmp_path):
    m = LanceMemory(embed_fn=fake_embed, path=str(tmp_path / "mem"))
    fid = m.remember("the wifi password is hunter2", {"kind": "fact"})
    assert isinstance(fid, str)
    hits = m.recall("wifi password", k=1)
    assert hits and "hunter2" in hits[0].text


def test_persists_across_restart(tmp_path):
    p = str(tmp_path / "mem")
    LanceMemory(fake_embed, p).remember("dentist on tuesday", {})
    hits = LanceMemory(fake_embed, p).recall("dentist", k=1)  # new instance = restart
    assert hits and "tuesday" in hits[0].text
