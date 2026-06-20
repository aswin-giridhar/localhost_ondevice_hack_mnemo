# src/mnemo/trace.py
import json
import os


class TraceLog:
    def __init__(self, path: str):
        self.path = path

    def _next_seq(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path) as f:
            return sum(1 for _ in f)

    def log(self, turn: dict) -> None:
        turn = {"seq": self._next_seq(), **turn}
        with open(self.path, "a") as f:
            f.write(json.dumps(turn) + "\n")

    def turns(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]
