# src/mnemo/vision.py
import os
import time
from pathlib import Path

import ollama
from PIL import Image, ExifTags

# Both qwen3-vl:2b and moondream fit 4GB at q4. Default to the tag that is
# actually pulled locally; override with MNEMO_VLM_MODEL.
VLM_MODEL = os.getenv("MNEMO_VLM_MODEL", "qwen3-vl:2b")
CAMERA_INDEX = int(os.getenv("MNEMO_CAMERA_INDEX", "0"))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
_MIN_DIM = 32  # below this it's an icon/thumbnail, not a real capture


def capture(out_path: str | None = None, index: int | None = None) -> str:
    """Grab one frame from the device camera and write it to disk.

    Uses OpenCV VideoCapture on the local webcam. This is the laptop-camera
    path; the phone (S24 Ultra) camera is the other device-camera path and
    POSTs frames to /photo over LAN. Raises RuntimeError with a clear message if
    no camera device is reachable (e.g. inside WSL2, where the host webcam is
    not exposed without usbipd-win).
    """
    import cv2  # imported lazily so the module loads without a camera present

    idx = CAMERA_INDEX if index is None else index
    out_path = out_path or str(Path.cwd() / f"capture_{int(time.time())}.jpg")
    cam = cv2.VideoCapture(idx)
    try:
        if not cam.isOpened():
            raise RuntimeError(
                f"no camera at index {idx} (on WSL2 attach the USB camera with "
                "usbipd-win, or use the phone camera -> POST /photo over LAN)")
        ok, frame = cam.read()
        if not ok or frame is None:
            raise RuntimeError(f"camera at index {idx} opened but returned no frame")
        if not cv2.imwrite(out_path, frame):
            raise RuntimeError(f"failed to write captured frame to {out_path}")
    finally:
        cam.release()
    return out_path


def validate(img_path: str) -> dict:
    """Real on-device image-integrity check on the actual bytes (no SDK, no stub).

    `ok` is False for corrupt/undecodable/too-small files. `reasons` records the
    verdict and trust signals (camera EXIF distinguishes a genuine capture from a
    screenshot/generated image; webcam frames legitimately have none).
    """
    reasons: list[str] = []
    try:
        with Image.open(img_path) as im:
            im.verify()  # detects truncation/corruption (consumes the handle)
        with Image.open(img_path) as im:
            fmt, (w, h), exif = im.format, im.size, im.getexif()
    except Exception as e:
        return {"ok": False, "reasons": [f"undecodable/corrupt image: {e}"], "meta": {}}

    ok = True
    reasons.append(f"decoded OK as {fmt} {w}x{h}")
    if w < _MIN_DIM or h < _MIN_DIM:
        ok = False
        reasons.append(f"too small ({w}x{h} < {_MIN_DIM}px) — likely not a real capture")

    tags = {ExifTags.TAGS.get(k, k): v for k, v in dict(exif).items()} if exif else {}
    make, model = tags.get("Make"), tags.get("Model")
    when = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if make or model:
        reasons.append(f"camera EXIF present (make={make!r}, model={model!r})")
    else:
        reasons.append("no camera EXIF (webcam frame / screenshot / stripped) — not blocking")
    if when:
        reasons.append(f"capture timestamp: {when}")

    return {"ok": ok, "reasons": reasons,
            "meta": {"format": fmt, "width": w, "height": h,
                     "has_camera_exif": bool(make or model), "captured_at": when}}


def extract(img_path: str) -> str:
    resp = ollama.chat(model=VLM_MODEL, messages=[{
        "role": "user",
        "content": "Extract the key facts from this image in one line.",
        "images": [img_path]}])
    return resp["message"]["content"].strip()


def capture_and_ingest(mem, out_path: str | None = None) -> str:
    """Capture from the device camera, then validate+extract+remember it."""
    from . import tools

    path = capture(out_path)
    return tools.run("ingest_photo", {"path": path}, mem)


def ingest_folder(mem, folder: str) -> list[dict]:
    """Phone-as-camera fallback: ingest every image dropped into a folder, then
    move it to `.processed/` so it isn't ingested twice. Real I/O, no stub."""
    from . import tools

    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    done = p / ".processed"
    done.mkdir(exist_ok=True)
    results = []
    for f in sorted(p.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _IMAGE_EXTS:
            continue
        res = tools.run("ingest_photo", {"path": str(f)}, mem)
        results.append({"file": f.name, "result": res})
        f.rename(done / f.name)
    return results
