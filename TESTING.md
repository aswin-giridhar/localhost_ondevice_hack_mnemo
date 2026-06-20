# Testing Mnemo on Windows (native)

Mnemo was built on a WSL2 box with **no webcam and no microphone**, so the camera
and voice paths could only be skipped there. Run it on a **native Windows machine**
to exercise those for real. All commands assume **PowerShell**, run from the repo
root after cloning.

---

## Quick paste-in prompt for a Windows Claude Code instance

Paste this to have an agent set up, run, and test everything:

```
You are setting up and testing "Mnemo", a fully-offline on-device personal assistant,
on this native Windows machine (which has a real webcam and mic). Repo:
https://github.com/aswin-giridhar/localhost_ondevice_hack_mnemo.git

Read the repo's README and the agent.py / cognee_backend.py comments to understand it
first. It's a Python tool-calling agent over a local SLM (Ollama), with a LanceDB
memory backend (default) and an optional Cognee graph backend, plus voice
(faster-whisper), vision (qwen3-vl), a FastAPI UI with an OFFLINE badge, an
Overmind-style improve loop, and a sandboxed self-tool feature.

Do the following and report honestly what passes/fails (don't fake anything):

1. PREREQS
   - Confirm Python 3.11+ and that Ollama for Windows is running
     (curl http://localhost:11434/api/tags).
   - Pull models while online:
       ollama pull hf.co/LiquidAI/LFM2-1.2B-Tool-GGUF
       ollama pull phi4-mini
       ollama pull qwen3-vl:2b
       ollama pull nomic-embed-text
   - git clone the repo, cd in, create a venv, pip install -r requirements.txt.
   - Warm Whisper once while online:
       python -c "from faster_whisper import WhisperModel; WhisperModel('base',device='cpu',compute_type='int8')"

2. FAST TESTS (no models): $env:PYTHONPATH="src"; python -m pytest -q   -> expect 32 passed.

3. REAL LIVE TESTS (Ollama + models): python -m pytest tests/integration -v -s
   - The two tests skipped on the WSL2 box should now RUN here:
       * test_live_camera_capture -> real webcam (set $env:MNEMO_CAMERA_INDEX if not 0)
       * test_live_voice_transcribe_real_speech -> install eSpeak NG for Windows and put
         espeak-ng on PATH, OR test voice manually via the UI with a recorded .wav.
   - If cognee isn't installed, add: -k "not cognee".

4. RUN THE APP + DEMO
   - $env:PYTHONPATH="src"; $env:MNEMO_SYNTH_MODEL="phi4-mini"
   - python -m uvicorn mnemo.server:app --host 0.0.0.0 --port 8000  ; open http://localhost:8000
   - 60-second story: (a) type "Remember my landlord is Mr Okafor" -> send;
     (b) stop+restart the server; (c) ask "Who is my landlord?" -> answers Mr Okafor;
     (d) Wi-Fi off / airplane mode -> badge flips to OFFLINE ✓ -> ask again, still works;
     (e) click "📷 Capture" -> real webcam frame -> VLM describes it -> remembered;
     (f) upload a short .wav via the voice input -> transcribed -> acted on.

5. REPORT a table of every test/step PASS/FAIL/SKIP with the real reason. For camera
   and voice, confirm whether they now work on real hardware. Do not edit code to force
   tests green unless it's a genuine bug; show the actual error if something fails.
```

---

## 1. Prerequisites

1. **Python 3.11+** (`python --version`).
2. **Ollama for Windows** (https://ollama.com) running as a service; verify with
   `curl http://localhost:11434/api/tags`.
3. **Pull models while online** (the demo itself runs offline):
   ```powershell
   ollama pull hf.co/LiquidAI/LFM2-1.2B-Tool-GGUF   # tool-caller (chain #1)
   ollama pull phi4-mini                            # reasoning + synthesis + cognee
   ollama pull qwen3-vl:2b                           # vision (VLM)
   ollama pull nomic-embed-text                      # embeddings
   ```
4. **Clone + install:**
   ```powershell
   git clone https://github.com/aswin-giridhar/localhost_ondevice_hack_mnemo.git
   cd localhost_ondevice_hack_mnemo
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   (Cognee + transformers are heavy and only needed for the Cognee backend in §6;
   comment them out of `requirements.txt` for a faster first install.)
5. **Warm Whisper while online:**
   ```powershell
   python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('whisper ready')"
   ```

## 2. Fast test suite (no models)

```powershell
$env:PYTHONPATH="src"; python -m pytest -q
```
Expected: **32 passed**.

## 3. Real end-to-end suite (Ollama + models)

```powershell
$env:PYTHONPATH="src"; python -m pytest tests/integration -v -s
```
On Windows the two WSL2 skips should now run:
- `test_live_camera_capture` — real webcam (`$env:MNEMO_CAMERA_INDEX="1"` if not 0).
- `test_live_voice_transcribe_real_speech` — needs `espeak-ng` on PATH (eSpeak NG for
  Windows) to synthesize speech; otherwise it SKIPs (test voice via the UI instead).
- `test_live_cognee_roundtrip_restart` — only if cognee is installed (§6), else use
  `-k "not cognee"`.

## 4. Launch the app

```powershell
$env:PYTHONPATH="src"; $env:MNEMO_SYNTH_MODEL="phi4-mini"
python -m uvicorn mnemo.server:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 (Allow the Windows Firewall prompt so the phone can reach
`http://<laptop-ip>:8000` over LAN).

## 5. The 60-second demo

1. Type `Remember my landlord is Mr Okafor` → send.
2. Stop the server (Ctrl+C) and start it again → proves persistence.
3. Ask `Who is my landlord?` → answers **Mr Okafor**.
4. Turn off Wi-Fi / airplane mode → badge flips to **OFFLINE ✓** → ask again, still works.
5. Click **📷 Capture** → real webcam frame → VLM describes it → remembered.
6. Upload a short `.wav` via the **voice** input → transcribed → acted on.

## 6. Optional: Cognee graph-memory backend

```powershell
pip install cognee transformers
$env:MNEMO_MEMORY_BACKEND="cognee"
```
Runs fully offline (config in `src/mnemo/cognee_backend.py`). `cognify` does per-write
graph extraction and is slow on CPU — `lance` (default) is the recommended demo path.

## 7. Windows gotchas

- PowerShell env vars use `$env:NAME="value"` (not `set`/`export`).
- Ollama must be running before the integration tests / the app.
- First Whisper / VLM call downloads a model — do it once online.
- Webcam index defaults to 0; override with `MNEMO_CAMERA_INDEX`.
- `scripts/net_guard.sh` is Linux-only; on Windows prove offline by toggling Wi-Fi —
  the live `/health` badge reflects it.

## 8. Already verified (WSL2) vs. verify-on-Windows

| Path | WSL2 build box | Your Windows box |
|------|----------------|------------------|
| Memory round-trip + restart | ✅ | re-confirm |
| Agent recall (LFM2 + phi4-mini synth) | ✅ | re-confirm |
| Vision VLM extract | ✅ | re-confirm |
| Offline-blocked turn | ✅ | re-confirm |
| Self-tool authoring | ✅ | re-confirm |
| Cognee offline round-trip | ✅ | optional |
| **Camera capture** | ⏭ no camera in WSL2 | **verify here** |
| **Voice transcription** | ⏭ no TTS in WSL2 | **verify here** |
