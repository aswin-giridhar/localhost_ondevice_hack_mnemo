# src/mnemo/cognee_backend.py
# Cognee configured for LOCAL-ONLY operation. These env vars MUST be set before
# `import cognee` (Cognee reads them at import time). Defaults point every LLM /
# embedding / store call at localhost Ollama + on-disk stores so nothing ever
# leaves the machine. This is the #1 offline trap: stock Cognee defaults to
# OpenAI for entity extraction + embeddings.
import os

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "phi4-mini")
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434/v1")
os.environ.setdefault("LLM_API_KEY", "ollama")  # dummy; never leaves localhost
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_ENDPOINT", "http://localhost:11434/v1")
os.environ.setdefault("VECTOR_DB_PROVIDER", "lancedb")
os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "kuzu")

import asyncio

from .memory import Fact


class CogneeMemory:
    """Cognee-local adapter implementing the same Memory protocol as LanceMemory.

    Cognee's API is async; we wrap it onto the sync remember/recall/all_facts
    interface so the agent never knows which backend is live.
    (API names may shift by Cognee version; the 45-min spike covers reconciling
    `cognee.add`/`cognify`/`search` to the installed version's quickstart.)
    """

    def __init__(self, settings):
        import cognee

        self.cognee = cognee
        self._loop = asyncio.new_event_loop()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def remember(self, text: str, meta: dict | None = None) -> str:
        self._run(self.cognee.add(text))
        self._run(self.cognee.cognify())
        return text[:24]

    def recall(self, query: str, k: int = 5) -> list[Fact]:
        res = self._run(self.cognee.search(query_text=query))
        return [Fact(id=str(i), text=str(r), meta={}) for i, r in enumerate(res[:k])]

    def all_facts(self) -> list[Fact]:
        return self.recall("", k=50)
