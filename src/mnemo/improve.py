# src/mnemo/improve.py
# Overmind-style loop: mine failed+corrected turns from the trace log, turn them
# into few-shot guidance, and fold that guidance into the agent's system prompt
# live. Fully offline — it only reads local traces and edits a string.
#
# Task 12 (TOP STRETCH, sandboxed): the agent writes its own tool. The local
# model emits a small pure function (args, mem) -> result; we validate it with an
# AST allowlist BEFORE it ever runs (no imports, no dunder access, no
# eval/exec/open/...), exec it into a namespace with a tiny builtins whitelist,
# then register it at runtime. Two layers (static allowlist + restricted
# builtins) keep "self-coding" demoable without handing the model a shell.
import ast
from pathlib import Path

from . import agent

# Builtins a learned tool may use. Deliberately excludes anything that can reach
# the filesystem, network, imports, or the interpreter internals.
_SAFE_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "sorted": sorted,
    "sum": sum, "min": min, "max": max, "range": range, "enumerate": enumerate,
    "any": any, "all": all, "abs": abs, "round": round, "reversed": reversed,
    "zip": zip, "map": map, "filter": filter,
    "True": True, "False": False, "None": None,
}
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "exit", "quit", "help",
    "breakpoint", "memoryview", "__builtins__",
}
# Method names that enable string-format sandbox escapes
# (e.g. "{0.__class__}".format(obj)) or type-graph traversal.
_FORBIDDEN_ATTRS = {"format", "format_map", "mro", "subclasses"}


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


# --- Task 12: agent-authored tools (sandboxed) -----------------------------

def validate_tool_code(code: str, func_name: str) -> None:
    """Raise ValueError unless `code` is exactly one safe function `func_name`.

    Safety = static AST allowlist applied BEFORE execution: a single top-level
    function with the expected name, no imports, no dunder attribute access, no
    f-strings/.format (string-format escapes), no use of forbidden
    builtins/calls.

    NOTE: this is defense-in-depth, NOT a true security boundary. In-process
    Python sandboxing cannot be made fully escape-proof. The feature is OFF by
    default (MNEMO_ENABLE_SELFTOOL) and intended for a single-user offline demo;
    for untrusted input, run generated code in an OS-level sandbox (nsjail /
    firejail / gVisor / a no-network read-only container) instead.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"learned tool does not parse: {e}") from e
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("learned tool must be exactly one function definition")
    if tree.body[0].name != func_name:
        raise ValueError(
            f"function name {tree.body[0].name!r} != expected {func_name!r}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed in learned tools")
        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            raise ValueError("f-strings are not allowed in learned tools")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("dunder attribute access is not allowed")
            if node.attr in _FORBIDDEN_ATTRS:
                raise ValueError(f"attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"use of {node.id!r} is not allowed")


def load_tool(code: str, func_name: str):
    """Validate then exec `code` in a restricted namespace; return the callable."""
    validate_tool_code(code, func_name)
    # Fresh copy per exec so one tool can't mutate builtins seen by later tools.
    ns: dict = {"__builtins__": dict(_SAFE_BUILTINS)}
    exec(compile(code, "<learned_tool>", "exec"), ns)  # AST-validated above
    return ns[func_name]


def _extract_code(text: str) -> str:
    """Pull a python function body out of a model reply (handles ``` fences)."""
    if "```" in text:
        block = text.split("```", 2)[1]
        if block.lstrip().startswith("python"):
            block = block.lstrip()[len("python"):]
        return block.strip() + "\n"
    return text.strip() + "\n"


def propose_tool(name: str, description: str, chat_fn) -> str:
    """Ask the local model for a pure function implementing the tool."""
    prompt = (
        f"Write a single Python function named {name} with the exact signature "
        f"`def {name}(args, mem):`. It should: {description}. "
        "`args` is a dict of string arguments; `mem` is the memory store "
        "(mem.remember(text, meta), mem.recall(query, k), mem.all_facts()). "
        "Return a string. Use ONLY plain Python and these builtins: "
        "len,str,int,float,list,dict,sorted,sum,min,max,range,enumerate. "
        "NO imports, NO file/network access, NO eval/exec. "
        "Reply with ONLY the function in a ```python code block."
    )
    out = chat_fn([
        {"role": "system", "content": "You write tiny, safe, pure Python tools."},
        {"role": "user", "content": prompt},
    ])
    return _extract_code(out["content"])


def propose_and_register(name, description, path, chat_fn, registrar=None) -> str:
    """End-to-end: generate -> validate -> persist -> register a learned tool.

    Validation happens BEFORE persistence/registration, so unsafe generated code
    is never written to disk or made callable.
    """
    code = propose_tool(name, description, chat_fn)
    fn = load_tool(code, name)  # validates; raises ValueError if unsafe
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write("\n\n" + code)
    from . import tools
    (registrar or tools.register)(name, fn)
    return code
