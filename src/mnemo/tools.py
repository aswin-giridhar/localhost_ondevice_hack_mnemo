# src/mnemo/tools.py
SCHEMAS = [
    {"type": "function", "function": {
        "name": "remember",
        "description": "Store a fact the user wants kept for later.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the fact to store"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Search stored memory for relevant facts.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_facts",
        "description": "List everything currently remembered.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "ingest_photo",
        "description": "Validate and read a photo file, then remember what it shows.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "capture_photo",
        "description": "Take a photo with the device camera, then remember what it shows.",
        "parameters": {"type": "object", "properties": {}}}},
]


# Runtime-registered tools the agent wrote for itself (Task 12, sandboxed).
# Each learned tool is a callable (args: dict, mem) -> any.
LEARNED: dict = {}


def register(name: str, func) -> None:
    LEARNED[name] = func


def _is_duplicate(mem, text: str) -> bool:
    """True if an (almost) identical fact is already stored — dedupe before write
    so a flaky small model calling remember twice doesn't bloat memory."""
    norm = " ".join(text.lower().split())
    try:
        nearby = mem.recall(text, k=5)
    except Exception:
        nearby = []
    return any(" ".join(f.text.lower().split()) == norm for f in nearby)


def run(name: str, args: dict, mem) -> str:
    if name in LEARNED:
        return str(LEARNED[name](args, mem))
    if name == "remember":
        text = args["text"]
        if _is_duplicate(mem, text):
            return f"already known: {text}"
        mem.remember(text, {"kind": "fact"})
        return f"stored: {text}"
    if name == "recall":
        hits = mem.recall(args.get("query", ""), k=5)
        return "; ".join(f.text for f in hits) or "(nothing relevant found)"
    if name == "list_facts":
        return "; ".join(f.text for f in mem.all_facts()) or "(memory empty)"
    if name == "ingest_photo":
        from . import vision
        v = vision.validate(args["path"])
        text = vision.extract(args["path"])
        mem.remember(f"[photo] {text}", {"kind": "photo", "validated": v["ok"]})
        reasons = "; ".join(v.get("reasons", []))
        return f"validated={v['ok']} ({reasons}); extracted: {text}"
    if name == "capture_photo":
        from . import vision
        return vision.capture_and_ingest(mem)
    return f"(unknown tool {name})"
