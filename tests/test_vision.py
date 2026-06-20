# tests/test_vision.py
# Unit tests for vision.validate use REAL image bytes (generated with PIL), not
# mocks. Real camera capture and real VLM extraction are exercised in
# tests/integration/test_live.py (they need a camera / Ollama).
from PIL import Image

from mnemo import vision


def _make_jpeg(path, size=(120, 90), color=(200, 120, 40)):
    Image.new("RGB", size, color).save(path, "JPEG")
    return str(path)


def test_validate_accepts_real_image(tmp_path):
    p = _make_jpeg(tmp_path / "real.jpg")
    v = vision.validate(p)
    assert v["ok"] is True
    assert v["meta"]["width"] == 120 and v["meta"]["height"] == 90
    assert any("decoded OK" in r for r in v["reasons"])


def test_validate_rejects_corrupt(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff not a real jpeg")
    v = vision.validate(str(bad))
    assert v["ok"] is False
    assert any("undecodable" in r or "corrupt" in r for r in v["reasons"])


def test_validate_rejects_too_small(tmp_path):
    p = _make_jpeg(tmp_path / "tiny.jpg", size=(8, 8))
    v = vision.validate(p)
    assert v["ok"] is False
    assert any("too small" in r for r in v["reasons"])
