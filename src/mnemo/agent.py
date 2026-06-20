# src/mnemo/agent.py
from . import model, tools

SYSTEM = (
    "You are Mnemo, a fully-offline personal assistant. "
    "Use the remember tool when the user tells you something to keep. "
    "Use recall to look things up before answering questions about the user. "
    "Be concise."
)


class Agent:
    def __init__(self, mem, trace=None, max_steps: int = 5):
        self.mem = mem
        self.trace = trace
        self.max_steps = max_steps

    def handle(self, user_input: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}]
        # Spec §4: proactively recall relevant facts and inject them so recall
        # doesn't depend on the small model deciding to call the recall tool.
        try:
            recalled = self.mem.recall(user_input, k=5)
        except Exception:
            recalled = []
        if recalled:
            facts = "\n".join(f"- {f.text}" for f in recalled)
            messages.append({
                "role": "system",
                "content": "Relevant things you remember about the user:\n" + facts})
        messages.append({"role": "user", "content": user_input})
        calls_made = []
        for _ in range(self.max_steps):
            out = model.chat(messages, tools=tools.SCHEMAS)
            if out["tool_calls"]:
                # Record the assistant's tool-call turn BEFORE the observations,
                # else the model can't see it already called the tool and just
                # re-emits the same call every step until max_steps.
                messages.append({
                    "role": "assistant",
                    "content": out["content"],
                    "tool_calls": [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in out["tool_calls"]],
                })
                for tc in out["tool_calls"]:
                    try:
                        result = tools.run(tc["name"], tc["arguments"], self.mem)
                    except Exception as e:  # tool error -> observation, keep going
                        result = f"(tool error: {e})"
                    calls_made.append(
                        {"tool": tc["name"], "args": tc["arguments"], "result": result})
                    messages.append(
                        {"role": "tool", "content": result, "tool_name": tc["name"]})
                continue
            reply = out["content"] or "(no response)"
            if self.trace:
                self.trace.log(
                    {"input": user_input, "calls": calls_made, "reply": reply, "ok": True})
            return reply
        reply = "(stopped after max steps)"
        if self.trace:
            self.trace.log(
                {"input": user_input, "calls": calls_made, "reply": reply, "ok": False})
        return reply
