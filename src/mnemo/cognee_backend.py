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
# STATUS: routing is confirmed local (no phone-home), but a green round-trip+
# restart gate was NOT achieved within the time/cost box -> per the plan's abort
# rule the shipped default stays MNEMO_MEMORY_BACKEND=lance. This file is a
# near-complete, UNVERIFIED swap; finish the gate before flipping the default.
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "phi4-mini")
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434/v1")
os.environ.setdefault("LLM_API_KEY", "ollama")  # dummy; never leaves localhost
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_ENDPOINT", "http://localhost:11434/v1")
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
