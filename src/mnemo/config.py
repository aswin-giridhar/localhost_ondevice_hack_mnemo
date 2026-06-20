import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    llm_models: list[str] = field(
        default_factory=lambda: ["phi4-mini", "qwen3:4b", "llama3.2:3b"]
    )
    embed_model: str = "nomic-embed-text"
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MNEMO_DATA_DIR", "data"))
    )
    memory_backend: str = field(
        default_factory=lambda: os.getenv("MNEMO_MEMORY_BACKEND", "lance")
    )  # the noon spike flips this to "cognee" if local Cognee works offline


def load_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s


def data_path(name: str, s: "Settings | None" = None) -> Path:
    s = s or load_settings()
    return s.data_dir / name


SETTINGS = load_settings()
