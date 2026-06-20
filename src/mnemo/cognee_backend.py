# src/mnemo/cognee_backend.py
# Cognee configured for LOCAL-ONLY operation. These env vars MUST be set before
# `import cognee` (Cognee reads them at import time). Defaults point every LLM /
# embedding / store call at localhost Ollama + on-disk stores so nothing ever
# leaves the machine. This is the #1 offline trap: stock Cognee defaults to
# OpenAI for entity extraction + embeddings.
import os

# --- Cognee local-only config (verified against cognee 1.1.3, 2026-06-20) ----
# Spike findings — these are the env vars cognee 1.1.3 actually needs to run
# fully offline. The naive 3-var config in the plan is NOT enough on 1.1.3:
#   1. Ollama EMBEDDINGS also require EMBEDDING_DIMENSIONS + HUGGINGFACE_TOKENIZER
#      (used for token counting), else cognee raises a LLMConfig ValidationError.
#   2. cognify() does structured entity extraction via BAML, which defaults to
#      OpenAI (gpt-5-mini). Without BAML_LLM_* pointed at Ollama, cognify phones
#      home. (Confirmed: with these set, the OpenAI sentinel key was never used.)
#   3. cognee runs a 30s LLM connection test at startup; phi4-mini cold-load on a
#      4GB box exceeds it -> COGNEE_SKIP_CONNECTION_TEST=true (warm the model
#      first).
#   4. HUGGINGFACE_TOKENIZER makes cognee load the tokenizer via the
#      `transformers` lib, which is NOT a cognee dependency -> `pip install
#      transformers`.
#   5. cognee's OllamaEmbeddingEngine POSTs to EMBEDDING_ENDPOINT verbatim and
#      reads data["embeddings"][0] -> the endpoint must be Ollama's NATIVE embed
#      API: "http://localhost:11434/api/embed" (NOT the /v1 OpenAI path, which
#      404s). Fixed below.
#   6. recall() must use SearchType.CHUNKS (raw vector retrieval), not the
#      default GRAPH_COMPLETION: the latter makes the small model generate an
#      answer via BAML and fails structured-output validation on phi4-mini.
# STATUS (2026-06-20): VERIFIED offline end-to-end -- add -> cognify -> search
# (CHUNKS) returned the stored fact with the OpenAI sentinel never used (no
# phone-home). Cognee-local works. The shipped DEFAULT still stays
# MNEMO_MEMORY_BACKEND=lance because cognify() is slow on CPU (per-write graph
# extraction) and lance is the fast, proven demo path; set
# MNEMO_MEMORY_BACKEND=cognee to use this graph-memory backend. Requires:
# `pip install cognee transformers` + ollama models phi4-mini + nomic-embed-text.
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "phi4-mini")
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434/v1")
os.environ.setdefault("LLM_API_KEY", "ollama")  # dummy; never leaves localhost
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_ENDPOINT", "http://localhost:11434/api/embed")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ.setdefault("HUGGINGFACE_TOKENIZER", "nomic-ai/nomic-embed-text-v1.5")
os.environ.setdefault("BAML_LLM_PROVIDER", "ollama")
os.environ.setdefault("BAML_LLM_MODEL", "phi4-mini")
os.environ.setdefault("BAML_LLM_ENDPOINT", "http://localhost:11434/v1")
os.environ.setdefault("BAML_LLM_API_KEY", "ollama")
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
os.environ.setdefault("VECTOR_DB_PROVIDER", "lancedb")
os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "kuzu")

import asyncio
import json
import threading
import uuid
from pathlib import Path

from .memory import Fact


class CogneeMemory:
    """Cognee-local adapter implementing the same Memory protocol as LanceMemory.

    Cognee's API is async; we wrap it onto the sync remember/recall/all_facts
    interface so the agent never knows which backend is live.
      - recall(): real semantic retrieval via Cognee CHUNKS search.
      - remember(): adds to Cognee + runs the slow graph-extraction (cognify) in
        a BACKGROUND thread so the agent/UI never blocks on it.
      - all_facts(): reads a write-through JSON manifest of everything stored
        (Cognee's vector engine has no get-all primitive; the manifest is a real,
        persisted record — exact and restart-safe).
    """

    def __init__(self, settings):
        import cognee

        self.cognee = cognee
        self._loop = asyncio.new_event_loop()
        self._cognify_lock = threading.Lock()
        self._pending: list[threading.Thread] = []
        self._manifest = Path(settings.data_dir) / "cognee_facts.jsonl"
        self._manifest.parent.mkdir(parents=True, exist_ok=True)

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def remember(self, text: str, meta: dict | None = None) -> str:
        fid = uuid.uuid4().hex
        self._run(self.cognee.add(text))
        # Write-through manifest: exact record of what was stored, for all_facts.
        with open(self._manifest, "a") as f:
            f.write(json.dumps({"id": fid, "text": text, "meta": meta or {}}) + "\n")

        # cognify (graph extraction) is slow on CPU -> run it off the hot path.
        def _bg():
            with self._cognify_lock:  # serialize concurrent cognify runs
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self.cognee.cognify())
                finally:
                    loop.close()

        t = threading.Thread(target=_bg, daemon=True)
        t.start()
        self._pending.append(t)
        return fid

    def flush(self, timeout: float | None = None) -> None:
        """Block until all background cognify runs finish (tests / pre-recall)."""
        for t in list(self._pending):
            t.join(timeout)
        self._pending = [t for t in self._pending if t.is_alive()]

    def recall(self, query: str, k: int = 5) -> list[Fact]:
        # CHUNKS = raw vector retrieval (not GRAPH_COMPLETION, which makes the
        # small model generate an answer via BAML and is flaky on 3.8B models).
        from cognee.modules.search.types import SearchType

        res = self._run(
            self.cognee.search(query_text=query, query_type=SearchType.CHUNKS, top_k=k))
        facts: list[Fact] = []
        for group in res or []:
            items = group.get("search_result", []) if isinstance(group, dict) else []
            for rec in items:
                if not isinstance(rec, dict):
                    continue
                text = rec.get("text") or rec.get("content")
                if text:
                    facts.append(Fact(id=str(rec.get("id", "")), text=str(text), meta={}))
        return facts[:k]

    def all_facts(self) -> list[Fact]:
        if not self._manifest.exists():
            return []
        out = []
        with open(self._manifest) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out.append(Fact(id=r["id"], text=r["text"], meta=r.get("meta", {})))
        return out
