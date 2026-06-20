# src/mnemo/server.py
import json
import socket
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config
from .agent import Agent
from .memory import get_memory
from .trace import TraceLog

SETTINGS = config.load_settings()
MEM = get_memory(SETTINGS)
TRACE = TraceLog(str(config.data_path("traces.jsonl", SETTINGS)))
AGENT = Agent(MEM, trace=TRACE)
app = FastAPI()


class ChatIn(BaseModel):
    message: str


def _is_offline() -> bool:
    # Per-call timeout only — never mutate the process-wide default socket
    # timeout (that would also throttle the Ollama / LanceDB connections).
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1).close()
        return False
    except OSError:
        return True


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).resolve().parents[2].joinpath("web/index.html").read_text()


@app.post("/chat")
def chat(inp: ChatIn):
    return {"reply": AGENT.handle(inp.message)}


@app.get("/facts")
def facts():
    return {"facts": [f.text for f in MEM.all_facts()]}


@app.get("/health")
def health():
    return {"offline": _is_offline()}


@app.post("/voice")
async def voice_in(file: UploadFile):
    from . import voice

    tmp = config.data_path("_last.wav", SETTINGS)
    tmp.write_bytes(await file.read())
    text = voice.transcribe(str(tmp))
    return {"transcript": text, "reply": AGENT.handle(text)}


@app.post("/photo")
async def photo_in(file: UploadFile):
    from . import tools

    p = config.data_path("_last.jpg", SETTINGS)
    p.write_bytes(await file.read())
    return {"result": tools.run("ingest_photo", {"path": str(p)}, MEM)}


@app.post("/feedback")
def feedback(inp: dict):
    rows = TRACE.turns()
    if rows:
        rows[-1]["correction"] = inp.get("correction", "")
        rows[-1]["ok"] = False
        with open(TRACE.path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return {"ok": True}


@app.post("/improve")
def improve_now():
    from . import improve

    ex = improve.analyze(TRACE.turns())
    improve.apply(ex)
    return {"learned": ex}
