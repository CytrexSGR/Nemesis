"""Nemesis configuration — Pydantic BaseSettings with env var support."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class NemesisConfig(BaseSettings):
    """Central configuration for Nemesis.

    All fields can be overridden via environment variables prefixed with NEMESIS_.
    Example: NEMESIS_DATA_DIR=/custom/path
    """

    model_config = {
        "env_prefix": "NEMESIS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # Data directory
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".nemesis")
    languages: list[str] = Field(default_factory=lambda: ["python"])
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules",
            "venv",
            ".venv",
            "__pycache__",
            ".git",
            "target",
            "dist",
            "build",
        ]
    )

    # Graph DB
    graph_backend: Literal["kuzu", "neo4j"] = "kuzu"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Vector DB
    vector_provider: Literal["openai", "local"] = "openai"
    vector_model: str = "text-embedding-3-small"
    openai_api_key: str = ""

    # Memory
    memory_auto_load_rules: bool = True
    memory_auto_learn: bool = True

    # Watcher
    watcher_enabled: bool = True
    watcher_debounce_ms: int = 500

    @property
    def graph_dir(self) -> Path:
        """Return the graph database directory path."""
        return self.data_dir / "graph"

    @property
    def vector_dir(self) -> Path:
        """Return the vector database directory path."""
        return self.data_dir / "vectors"

    @property
    def registry_path(self) -> Path:
        """Return the project registry file path."""
        return self.data_dir / "registry.json"
