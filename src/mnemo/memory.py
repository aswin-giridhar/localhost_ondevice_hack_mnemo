# src/mnemo/memory.py
import json
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import lancedb


@dataclass
class Fact:
    id: str
    text: str
    meta: dict = field(default_factory=dict)
    score: float = 0.0


class Memory(Protocol):
    def remember(self, text: str, meta: dict | None = None) -> str: ...
    def recall(self, query: str, k: int = 5) -> list[Fact]: ...
    def all_facts(self) -> list[Fact]: ...


class LanceMemory:
    """Fully-local fallback: LanceDB vector store + injected embed fn."""

    def __init__(self, embed_fn, path: str):
        self.embed = embed_fn
        self.db = lancedb.connect(path)
        self._table = None

    def _tbl(self, dim: int | None = None):
        if self._table is not None:
            return self._table
        if "facts" in self.db.table_names():
            self._table = self.db.open_table("facts")
        elif dim is not None:
            self._table = self.db.create_table(
                "facts",
                data=[{"id": "_seed", "text": "", "meta": "{}", "vector": [0.0] * dim}],
            )
            self._table.delete("id = '_seed'")
        return self._table

    def remember(self, text: str, meta: dict | None = None) -> str:
        vec = self.embed(text)
        fid = uuid.uuid4().hex
        self._tbl(dim=len(vec)).add(
            [{"id": fid, "text": text, "meta": json.dumps(meta or {}), "vector": vec}]
        )
        return fid

    def recall(self, query: str, k: int = 5) -> list[Fact]:
        tbl = self._tbl()
        if tbl is None:
            return []
        rows = tbl.search(self.embed(query)).limit(k).to_list()
        return [
            Fact(r["id"], r["text"], json.loads(r["meta"]), float(r.get("_distance", 0.0)))
            for r in rows
        ]

    def all_facts(self) -> list[Fact]:
        tbl = self._tbl()
        if tbl is None:
            return []
        return [
            Fact(r["id"], r["text"], json.loads(r["meta"]))
            for r in tbl.to_pandas().to_dict("records")
        ]


def get_memory(settings):
    from .model import embed

    if settings.memory_backend == "cognee":
        from .cognee_backend import CogneeMemory  # added in Task 4

        return CogneeMemory(settings)
    return LanceMemory(embed_fn=embed, path=str(settings.data_dir / "lance"))
