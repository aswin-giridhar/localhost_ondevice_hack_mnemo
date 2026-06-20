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
]


def run(name: str, args: dict, mem) -> str:
    if name == "remember":
        mem.remember(args["text"], {"kind": "fact"})
        return f"stored: {args['text']}"
    if name == "recall":
        hits = mem.recall(args.get("query", ""), k=5)
        return "; ".join(f.text for f in hits) or "(nothing relevant found)"
    if name == "list_facts":
        return "; ".join(f.text for f in mem.all_facts()) or "(memory empty)"
    if name == "ingest_photo":  # used in Task 9
        from . import vision
        v = vision.validate(args["path"])
        text = vision.extract(args["path"])
        mem.remember(f"[photo] {text}", {"kind": "photo", "validated": v["ok"]})
        return f"validated={v['ok']}; extracted: {text}"
    return f"(unknown tool {name})"
