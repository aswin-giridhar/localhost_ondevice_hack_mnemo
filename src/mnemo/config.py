import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # Fallback chain: LFM2 is a purpose-built 1.2B tool-caller (tiny VRAM,
    # SLM-default per the agentic-SLM survey); phi4-mini is the reasoning
    # fallback; the rest are whatever is already pulled locally.
    llm_models: list[str] = field(
        default_factory=lambda: os.getenv(
            "MNEMO_LLM_MODELS",
            # First entry is the exact Ollama tag of the pulled LFM2 tool-caller.
            "hf.co/LiquidAI/LFM2-1.2B-Tool-GGUF,phi4-mini,qwen3:4b,qwen3.5:0.8b",
        ).split(",")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("MNEMO_EMBED_MODEL", "nomic-embed-text")
    )
    # Optional stronger model for final-answer synthesis after tools run (routing:
    # LFM2 for tool calls, e.g. phi4-mini for synthesis). Empty -> no extra call.
    synth_model: str = field(
        default_factory=lambda: os.getenv("MNEMO_SYNTH_MODEL", "")
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MNEMO_DATA_DIR", "data"))
    )
    memory_backend: str = field(
        default_factory=lambda: os.getenv("MNEMO_MEMORY_BACKEND", "lance")
    )  # the noon spike flips this to "cognee" if local Cognee works offline
    # Task 12 (top stretch): allow the agent to author + run its own tools at
    # runtime. OFF by default — it executes model-generated (sandboxed) code.
    enable_selftool: bool = field(
        default_factory=lambda: os.getenv("MNEMO_ENABLE_SELFTOOL", "0") == "1"
    )


def load_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s


def data_path(name: str, s: "Settings | None" = None) -> Path:
    s = s or load_settings()
    return s.data_dir / name


SETTINGS = load_settings()
