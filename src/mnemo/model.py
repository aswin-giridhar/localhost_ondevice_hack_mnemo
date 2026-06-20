import json
import ollama
from .config import SETTINGS


def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Call the local model, walking the fallback chain on any failure.

    Returns {"content": str, "tool_calls": [{"name": str, "arguments": dict}]}.
    """
    last_err = None
    for tag in SETTINGS.llm_models:
        try:
            resp = ollama.chat(model=tag, messages=messages, tools=tools)
            msg = resp["message"]
            calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                args = fn["arguments"]
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                calls.append({"name": fn["name"], "arguments": args})
            return {"content": msg.get("content", ""), "tool_calls": calls}
        except Exception as e:  # model missing / OOM -> next fallback
            last_err = e
            continue
    raise RuntimeError(f"all models failed: {last_err}")


def embed(text: str) -> list[float]:
    return ollama.embeddings(model=SETTINGS.embed_model, prompt=text)["embedding"]
