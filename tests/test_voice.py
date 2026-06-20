# tests/test_voice.py
from unittest.mock import patch, MagicMock

from mnemo import voice


def test_transcribe_joins_segments(tmp_path):
    seg = MagicMock(text="remember the keys are by the door")
    fake = MagicMock()
    fake.transcribe.return_value = ([seg], None)
    with patch("mnemo.voice._model", fake):
        out = voice.transcribe(str(tmp_path / "x.wav"))
    assert "keys are by the door" in out
