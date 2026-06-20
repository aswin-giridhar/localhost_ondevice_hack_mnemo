# tests/test_vision.py
from unittest.mock import patch

from mnemo import vision


def test_validate_then_extract(tmp_path):
    img = tmp_path / "r.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    with patch("mnemo.vision._captur_validate", return_value={"ok": True, "reasons": []}):
        v = vision.validate(str(img))
    assert v["ok"] is True
    with patch("mnemo.vision.ollama.chat",
               return_value={"message": {"content": "Receipt: Tesco £4.20"}}):
        text = vision.extract(str(img))
    assert "Tesco" in text
