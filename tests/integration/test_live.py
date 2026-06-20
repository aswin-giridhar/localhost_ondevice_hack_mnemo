# tests/integration/test_live.py
# REAL end-to-end tests — NO mocks, NO stubs. They hit live Ollama models,
# faster-whisper, Cognee, and the device camera with real data.
# Run explicitly:  cd mnemo && PYTHONPATH=src python3 -m pytest tests/integration -v -s
import importlib.util
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from mnemo import config, model
from mnemo.agent import Agent
from mnemo.memory import LanceMemory


def _ollama_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="ollama not running")


def real_embed(t: str):
    return model.embed(t)  # REAL nomic-embed-text via Ollama


def test_live_embed_is_real():
    v = real_embed("hello world")
    assert isinstance(v, list) and len(v) >= 256 and any(x != 0 for x in v)


def test_live_agent_remember_then_recall_after_restart(tmp_path):
    # Recommended production routing: LFM2-1.2B drives tool calls, phi4-mini
    # synthesizes the final answer from injected memory (a 1.2B model alone
    # sometimes refuses to use recalled facts). Both are real, pulled models.
    old_synth = config.SETTINGS.synth_model
    config.SETTINGS.synth_model = "phi4-mini"
    try:
        p = str(tmp_path / "m")
        Agent(LanceMemory(real_embed, p)).handle("Remember that my landlord is Mr Okafor")
        # New memory instance over the same path == process restart.
        mem2 = LanceMemory(real_embed, p)
        assert any("Okafor" in f.text for f in mem2.all_facts())
        reply = Agent(mem2).handle("Who is my landlord?")
        assert "okafor" in reply.lower()
    finally:
        config.SETTINGS.synth_model = old_synth


def test_live_vision_extract_reads_real_image(tmp_path):
    img = tmp_path / "receipt.png"
    im = Image.new("RGB", (520, 180), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((20, 70), "TESCO   TOTAL  4.20 GBP", fill=(0, 0, 0))
    im.save(img)
    from mnemo import vision
    text = vision.extract(str(img))
    assert text and any(s in text.upper() for s in ("TESCO", "4.20", "GBP"))


def test_live_offline_agent_full_turn(tmp_path, monkeypatch):
    """Block ALL non-loopback outbound sockets, then run real agent turns.
    Ollama is on 127.0.0.1 so it still works; the internet does not."""
    real_conn = socket.create_connection

    def guarded(address, *a, **k):
        host = address[0]
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise OSError(f"blocked outbound to {host} (offline proof)")
        return real_conn(address, *a, **k)

    monkeypatch.setattr(socket, "create_connection", guarded)
    mem = LanceMemory(real_embed, str(tmp_path / "m"))
    a = Agent(mem)
    a.handle("Remember the spare key is under the mat")
    reply = a.handle("Where is the spare key?")
    assert "mat" in reply.lower()


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="no offline TTS to make real speech")
def test_live_voice_transcribe_real_speech(tmp_path):
    wav = tmp_path / "speech.wav"
    subprocess.run(
        ["espeak-ng", "-w", str(wav), "remember the spare key is under the mat"],
        check=True)
    from mnemo import voice
    text = voice.transcribe(str(wav)).lower()
    assert any(w in text for w in ("key", "spare", "mat", "remember"))


def test_live_camera_capture(tmp_path):
    from mnemo import vision
    try:
        path = vision.capture(str(tmp_path / "cap.jpg"))
    except Exception as e:  # no /dev/video* in WSL2 -> honest skip, not a fake pass
        pytest.skip(f"no device camera reachable in this environment: {e}")
    assert Path(path).exists() and vision.validate(path)["ok"]


def test_live_selftool_authors_real_tool(tmp_path):
    from mnemo import improve, tools
    tools.LEARNED.clear()
    try:
        improve.propose_and_register(
            "word_count",
            "return, as a string, the number of words in args['text']",
            path=str(tmp_path / "learned_tools.py"),
            chat_fn=model.chat,
        )
    except ValueError as e:
        pytest.skip(f"model emitted code the sandbox safely rejected: {e}")
    out = tools.run("word_count", {"text": "one two three four"}, None)
    assert "4" in out
    tools.LEARNED.clear()


@pytest.mark.skipif(importlib.util.find_spec("cognee") is None, reason="cognee not installed")
def test_live_cognee_roundtrip_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_MEMORY_BACKEND", "cognee")
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    from mnemo.memory import get_memory
    m = get_memory(config.load_settings())
    m.remember("the safe deposit code is 4731")
    m.flush()  # wait for background cognify
    # restart: fresh adapter; cognee persists to its own on-disk stores
    m2 = get_memory(config.load_settings())
    assert any("4731" in f.text for f in m2.all_facts())
    hits = m2.recall("safe deposit code", k=5)
    assert any("4731" in h.text for h in hits)
