# src/mnemo/voice.py
# faster-whisper on CPU keeps the 4GB GPU free for the LLM.
# The model is loaded LAZILY (not at import) so (a) importing this module never
# triggers a model download, and (b) the offline demo never hits the network at
# run time — the base model is warmed during the build. The test patches
# `mnemo.voice._model` directly, which short-circuits the lazy loader.
_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe(wav_path: str) -> str:
    segments, _ = _get_model().transcribe(wav_path)
    return " ".join(s.text.strip() for s in segments).strip()
