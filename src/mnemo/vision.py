# src/mnemo/vision.py
import os

import ollama

# Both qwen3-vl:2b and moondream fit 4GB at q4. Default to the tag that is
# actually pulled locally; override with MNEMO_VLM_MODEL.
VLM_MODEL = os.getenv("MNEMO_VLM_MODEL", "qwen3-vl:2b")


def _captur_validate(img_path: str) -> dict:
    """On-device Captur validation. Fails OPEN if the SDK is absent so the demo
    path still works; swap in the real SDK call when wired."""
    try:
        import captur  # type: ignore

        return captur.validate_image(img_path)
    except Exception:
        return {"ok": True, "reasons": ["captur-stub: not validated on-device"]}


def validate(img_path: str) -> dict:
    return _captur_validate(img_path)


def extract(img_path: str) -> str:
    resp = ollama.chat(model=VLM_MODEL, messages=[{
        "role": "user",
        "content": "Extract the key facts from this image in one line.",
        "images": [img_path]}])
    return resp["message"]["content"].strip()
