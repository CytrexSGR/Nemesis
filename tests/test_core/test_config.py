"""Tests for Nemesis configuration."""

from pathlib import Path

import pytest


def test_config_default_values():
    """Config has sensible defaults for all fields."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.languages == ["python"]
    assert config.ignore_patterns == [
        "node_modules",
        "venv",
        ".venv",
        "__pycache__",
        ".git",
        "target",
        "dist",
        "build",
    ]


def test_config_graph_defaults():
    """Graph config defaults to kuzu backend."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.graph_backend == "kuzu"


def test_config_vector_defaults():
    """Vector config defaults to OpenAI embeddings."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.vector_provider == "openai"
    assert config.vector_model == "text-embedding-3-small"


def test_config_memory_defaults():
    """Memory config has auto-load and auto-learn enabled."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.memory_auto_load_rules is True
    assert config.memory_auto_learn is True


def test_config_watcher_defaults():
    """Watcher config has sensible defaults."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.watcher_enabled is True
    assert config.watcher_debounce_ms == 500


def test_config_openai_api_key():
    """OpenAI API key can be set via environment variable."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig(openai_api_key="sk-test-key-123")
    assert config.openai_api_key == "sk-test-key-123"


def test_config_env_override(monkeypatch):
    """Config values can be overridden via NEMESIS_ prefixed env vars."""
    from nemesis.core.config import NemesisConfig

    monkeypatch.setenv("NEMESIS_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("NEMESIS_WATCHER_DEBOUNCE_MS", "1000")

    config = NemesisConfig()
    assert config.graph_backend == "neo4j"
    assert config.watcher_debounce_ms == 1000


def test_config_neo4j_uri():
    """Neo4j URI defaults to bolt://localhost:7687."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.neo4j_uri == "bolt://localhost:7687"


def test_data_dir_defaults_to_home_nemesis():
    """data_dir field defaults to ~/.nemesis."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.data_dir == Path.home() / ".nemesis"


def test_graph_dir():
    """graph_dir property returns data_dir / 'graph'."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.graph_dir == config.data_dir / "graph"


def test_vector_dir():
    """vector_dir property returns data_dir / 'vectors'."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.vector_dir == config.data_dir / "vectors"


def test_registry_path():
    """registry_path property returns data_dir / 'registry.json'."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.registry_path == config.data_dir / "registry.json"


def test_data_dir_from_env(monkeypatch):
    """data_dir can be overridden via NEMESIS_DATA_DIR env var."""
    monkeypatch.setenv("NEMESIS_DATA_DIR", "/tmp/custom-nemesis")

    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.data_dir == Path("/tmp/custom-nemesis")


def test_config_graph_backend_validation():
    """Graph backend must be 'kuzu' or 'neo4j'."""
    from pydantic import ValidationError

    from nemesis.core.config import NemesisConfig

    with pytest.raises(ValidationError):
        NemesisConfig(graph_backend="invalid")


def test_config_vector_provider_validation():
    """Vector provider must be 'openai' or 'local'."""
    from pydantic import ValidationError

    from nemesis.core.config import NemesisConfig

    with pytest.raises(ValidationError):
        NemesisConfig(vector_provider="invalid")
