# src/mnemo/improve.py
# Overmind-style loop: mine failed+corrected turns from the trace log, turn them
# into few-shot guidance, and fold that guidance into the agent's system prompt
# live. Fully offline — it only reads local traces and edits a string.
from . import agent


def analyze(traces: list[dict]) -> list[str]:
    out = []
    for t in traces:
        if not t.get("ok") and t.get("correction"):
            out.append(f"When the user says '{t['input']}', do: {t['correction']}")
    return out


def apply(examples: list[str]) -> str:
    if not examples:
        return agent.SYSTEM
    block = "\n".join(f"- {e}" for e in examples)
    agent.SYSTEM = agent.SYSTEM + "\n\nLearned from past corrections:\n" + block
    return agent.SYSTEM
