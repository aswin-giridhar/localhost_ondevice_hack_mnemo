# src/mnemo/agent.py
from . import config, model, tools

# Few-shot, directive system prompt. Small models (LFM2-1.2B) call tools more
# reliably with concrete examples and an explicit "don't repeat a call" rule.
SYSTEM = (
    "You are Mnemo, a fully-offline personal assistant.\n"
    "Rules:\n"
    "- When the user tells you a fact to keep, call the remember tool ONCE with that fact.\n"
    "- When the user asks about themselves, relevant facts are already provided above; "
    "answer from them. Only call recall if they are not enough.\n"
    "- Never repeat the same tool call. After a tool returns, give a short final answer.\n"
    "- Be concise.\n"
    "Examples:\n"
    "User: Remember my dentist is Dr Lee.\n"
    "Assistant: (calls remember text='dentist is Dr Lee') Saved — your dentist is Dr Lee.\n"
    "User: Who is my dentist?\n"
    "Assistant: (facts show 'dentist is Dr Lee') Your dentist is Dr Lee.\n"
)

_KNOWN_TOOLS = {s["function"]["name"] for s in tools.SCHEMAS}


class Agent:
    def __init__(self, mem, trace=None, max_steps: int = 5):
        self.mem = mem
        self.trace = trace
        self.max_steps = max_steps

    def _maybe_author_tool(self, name: str, args: dict) -> bool:
        """Mid-turn gap -> author the missing tool, if self-tool authoring is on."""
        if name in _KNOWN_TOOLS or name in tools.LEARNED:
            return False
        if not config.SETTINGS.enable_selftool:
            return False
        from . import improve
        try:
            improve.propose_and_register(
                name,
                f"tool the assistant tried to call with args {list(args)}",
                path=str(config.data_path("learned_tools.py")),
                chat_fn=model.chat,
            )
            return name in tools.LEARNED
        except Exception:
            return False

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
                    self._maybe_author_tool(tc["name"], tc["arguments"])
                    try:
                        result = tools.run(tc["name"], tc["arguments"], self.mem)
                    except Exception as e:  # tool error -> observation, keep going
                        result = f"(tool error: {e})"
                    calls_made.append(
                        {"tool": tc["name"], "args": tc["arguments"], "result": result})
                    messages.append(
                        {"role": "tool", "content": result, "tool_name": tc["name"]})
                continue
            reply = self._finalize(messages, out["content"], bool(calls_made))
            if self.trace:
                self.trace.log(
                    {"input": user_input, "calls": calls_made, "reply": reply, "ok": True})
            return reply
        reply = "(stopped after max steps)"
        if self.trace:
            self.trace.log(
                {"input": user_input, "calls": calls_made, "reply": reply, "ok": False})
        return reply

    def _finalize(self, messages, content: str, used_tools: bool) -> str:
        """Optionally route the final answer to a stronger synthesis model.

        Tool-calling stays on the chain (LFM2 leads), but a 1.2B model writes
        terse/empty final answers; if MNEMO_SYNTH_MODEL is set we regenerate the
        final reply with it after tools have run. Off by default -> no extra call.
        """
        synth = config.SETTINGS.synth_model
        if synth and used_tools:
            try:
                out = model.chat(messages, models=[synth])
                if out["content"].strip():
                    return out["content"]
            except Exception:
                pass
        return content or "(no response)"
