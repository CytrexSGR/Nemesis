> **Arbeitspaket A2** — Teil 2 von 3 des Project Setup Plans

# Project Setup: Config + CLI + Fixtures + Integrationstest

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the core infrastructure for Nemesis — Pydantic configuration, CLI skeleton, test fixtures, and a final integration test — building on the package structure from Arbeitspaket A1.

**Architecture:** Monorepo with a Python package (`nemesis/`) and a Rust crate (`nemesis-parse/`). The Python package uses maturin as build backend for the Rust extension. Embedded databases (Kuzu, LanceDB) keep deployment simple. Pydantic BaseSettings provides typed configuration with env var override support.

**Tech Stack:** Python 3.11+, Rust/PyO3/maturin, pytest, ruff, Click, Pydantic, GitHub Actions

**Dieses Paket enthält:** Task 4 (Config), Task 5 (CLI), Task 7 (Fixtures), Task 10 (Integrationstest)

---

### Task 4: Pydantic Config (nemesis/core/config.py)

**Files:**
- Create: `nemesis/core/config.py`
- Test: `tests/test_core/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_core/__init__.py
```

```python
# tests/test_core/test_config.py
"""Tests for Nemesis configuration."""

import os
from pathlib import Path

import pytest


def test_config_default_values():
    """Config has sensible defaults for all fields."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.project_name == "nemesis"
    assert config.project_root == Path(".")
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
    assert config.graph_path == Path(".nemesis/graph")


def test_config_vector_defaults():
    """Vector config defaults to OpenAI embeddings."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.vector_provider == "openai"
    assert config.vector_model == "text-embedding-3-small"
    assert config.vector_path == Path(".nemesis/vectors")


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

    monkeypatch.setenv("NEMESIS_PROJECT_NAME", "my-project")
    monkeypatch.setenv("NEMESIS_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("NEMESIS_WATCHER_DEBOUNCE_MS", "1000")

    config = NemesisConfig()
    assert config.project_name == "my-project"
    assert config.graph_backend == "neo4j"
    assert config.watcher_debounce_ms == 1000


def test_config_neo4j_uri():
    """Neo4j URI defaults to bolt://localhost:7687."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.neo4j_uri == "bolt://localhost:7687"


def test_config_data_dir():
    """data_dir property returns the .nemesis directory path."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig(project_root=Path("/tmp/myproject"))
    assert config.data_dir == Path("/tmp/myproject/.nemesis")


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
```

**Step 2: Run test to verify it fails**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_core/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nemesis.core.config'" or "ImportError"

**Step 3: Write minimal implementation**

```python
# nemesis/core/config.py
"""Nemesis configuration — Pydantic BaseSettings with env var support."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class NemesisConfig(BaseSettings):
    """Central configuration for Nemesis.

    All fields can be overridden via environment variables prefixed with NEMESIS_.
    Example: NEMESIS_PROJECT_NAME=my-project
    """

    model_config = {"env_prefix": "NEMESIS_"}

    # Project
    project_name: str = "nemesis"
    project_root: Path = Path(".")
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
    graph_path: Path = Path(".nemesis/graph")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Vector DB
    vector_provider: Literal["openai", "local"] = "openai"
    vector_model: str = "text-embedding-3-small"
    vector_path: Path = Path(".nemesis/vectors")
    openai_api_key: str = ""

    # Memory
    memory_auto_load_rules: bool = True
    memory_auto_learn: bool = True

    # Watcher
    watcher_enabled: bool = True
    watcher_debounce_ms: int = 500

    @property
    def data_dir(self) -> Path:
        """Return the .nemesis data directory path."""
        return self.project_root / ".nemesis"
```

**Step 4: Run test to verify it passes**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_core/test_config.py -v`
Expected: PASS (11 passed)

**Step 5: Commit**
```bash
git add nemesis/core/config.py tests/test_core/
git commit -m "feat: add Pydantic config with env var support and validation"
```

---

### Task 5: CLI Skeleton (nemesis/core/cli.py)

**Files:**
- Create: `nemesis/core/cli.py`
- Test: `tests/test_core/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_cli.py
"""Tests for Nemesis CLI skeleton."""

from click.testing import CliRunner


def test_cli_main_group():
    """The main CLI group exists and shows help."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Nemesis" in result.output


def test_cli_version_flag():
    """--version flag prints the version."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_index_command_exists():
    """The 'index' command is registered."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["index", "--help"])
    assert result.exit_code == 0
    assert "Index" in result.output or "index" in result.output.lower()


def test_cli_query_command_exists():
    """The 'query' command is registered."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["query", "--help"])
    assert result.exit_code == 0


def test_cli_watch_command_exists():
    """The 'watch' command is registered."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["watch", "--help"])
    assert result.exit_code == 0


def test_cli_serve_command_exists():
    """The 'serve' command is registered."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_core/test_cli.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nemesis.core.cli'" or "ImportError"

**Step 3: Write minimal implementation**

```python
# nemesis/core/cli.py
"""Nemesis CLI — command-line interface for indexing, querying, and serving."""

from __future__ import annotations

import click

from nemesis import __version__


@click.group()
@click.version_option(version=__version__, prog_name="Nemesis")
def main() -> None:
    """Nemesis — GraphRAG context engine for AI coding agents."""


@main.command()
@click.argument("path", default=".", type=click.Path(exists=False))
@click.option("--languages", "-l", multiple=True, help="Languages to index.")
def index(path: str, languages: tuple[str, ...]) -> None:
    """Index a project directory."""
    click.echo(f"Indexing {path}...")


@main.command()
@click.argument("query_text")
@click.option("--limit", "-n", default=10, help="Max results.")
def query(query_text: str, limit: int) -> None:
    """Query the code graph with natural language."""
    click.echo(f"Querying: {query_text}")


@main.command()
@click.argument("path", default=".", type=click.Path(exists=False))
def watch(path: str) -> None:
    """Watch a project directory for changes."""
    click.echo(f"Watching {path}...")


@main.command()
@click.option("--host", default="localhost", help="Server host.")
@click.option("--port", default=3333, help="Server port.")
def serve(host: str, port: int) -> None:
    """Start the MCP server."""
    click.echo(f"Starting MCP server on {host}:{port}...")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_core/test_cli.py -v`
Expected: PASS (6 passed)

**Step 5: Commit**
```bash
git add nemesis/core/cli.py tests/test_core/test_cli.py
git commit -m "feat: add Click CLI skeleton with index, query, watch, serve commands"
```

---

### Task 7: Test-Infrastruktur (conftest.py, fixtures)

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_conftest.py`

**Step 1: Write the failing test**

```python
# tests/test_conftest.py
"""Tests for test infrastructure — verify fixtures work."""

from pathlib import Path


def test_tmp_project_fixture(tmp_project):
    """tmp_project fixture creates a temporary project directory."""
    assert tmp_project.exists()
    assert tmp_project.is_dir()


def test_tmp_project_has_nemesis_dir(tmp_project):
    """tmp_project fixture creates a .nemesis subdirectory."""
    nemesis_dir = tmp_project / ".nemesis"
    assert nemesis_dir.exists()
    assert nemesis_dir.is_dir()


def test_nemesis_config_fixture(nemesis_config):
    """nemesis_config fixture returns a valid NemesisConfig."""
    from nemesis.core.config import NemesisConfig

    assert isinstance(nemesis_config, NemesisConfig)
    assert nemesis_config.project_root.exists()


def test_sample_python_file_fixture(sample_python_file):
    """sample_python_file fixture creates a Python file for testing."""
    assert sample_python_file.exists()
    assert sample_python_file.suffix == ".py"
    content = sample_python_file.read_text()
    assert "class Calculator" in content
    assert "def add" in content
```

**Step 2: Run test to verify it fails**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_conftest.py -v`
Expected: FAIL with "fixture 'tmp_project' not found"

**Step 3: Write minimal implementation**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared test fixtures for Nemesis."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with .nemesis subdirectory."""
    project_dir = tmp_path / "test-project"
    project_dir.mkdir()
    nemesis_dir = project_dir / ".nemesis"
    nemesis_dir.mkdir()
    return project_dir


@pytest.fixture
def nemesis_config(tmp_project: Path):
    """Create a NemesisConfig pointing to the tmp_project."""
    from nemesis.core.config import NemesisConfig

    return NemesisConfig(
        project_name="test-project",
        project_root=tmp_project,
        openai_api_key="sk-test-fake-key",
    )


@pytest.fixture
def sample_python_file(tmp_project: Path) -> Path:
    """Create a sample Python file for parser testing."""
    code = '''"""A sample module for testing."""

from typing import List


class Calculator:
    """A simple calculator class."""

    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return round(a + b, self.precision)

    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return round(a - b, self.precision)


def create_calculator(precision: int = 2) -> Calculator:
    """Factory function for Calculator."""
    return Calculator(precision=precision)


async def fetch_data(url: str) -> List[dict]:
    """Async function for testing async detection."""
    return []
'''
    file_path = tmp_project / "sample.py"
    file_path.write_text(code)
    return file_path
```

**Step 4: Run test to verify it passes**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_conftest.py -v`
Expected: PASS (4 passed)

**Step 5: Commit**
```bash
git add tests/
git commit -m "feat: add test infrastructure with project, config, and sample file fixtures"
```

---

### Task 10: Finaler Integrationstest — alles zusammen

**Files:**
- Test: `tests/test_integration_setup.py`

**Step 1: Write the failing test**

```python
# tests/test_integration_setup.py
"""Integration test for project setup — verifies everything works together."""

import subprocess
import sys
from pathlib import Path


def test_nemesis_version_consistency():
    """Version in __init__.py matches pyproject.toml."""
    import tomllib

    from nemesis import __version__

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    assert __version__ == pyproject["project"]["version"]


def test_all_subpackages_importable():
    """Every subpackage defined in the project can be imported."""
    subpackages = [
        "nemesis",
        "nemesis.core",
        "nemesis.core.config",
        "nemesis.core.cli",
        "nemesis.indexer",
        "nemesis.parser",
        "nemesis.graph",
        "nemesis.vector",
        "nemesis.memory",
        "nemesis.tools",
    ]
    for pkg in subpackages:
        __import__(pkg)


def test_config_creates_with_defaults():
    """NemesisConfig can be instantiated with all defaults."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.project_name == "nemesis"
    assert config.graph_backend == "kuzu"
    assert config.vector_provider == "openai"


def test_cli_entrypoint():
    """The nemesis CLI entry point is callable."""
    from click.testing import CliRunner

    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_pyproject_has_all_dependencies():
    """pyproject.toml lists all required runtime dependencies."""
    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    deps = pyproject["project"]["dependencies"]
    dep_names = [d.split(">=")[0].split("[")[0].strip() for d in deps]

    required = ["kuzu", "lancedb", "openai", "click", "pydantic", "pydantic-settings", "watchdog", "mcp"]
    for req in required:
        assert req in dep_names, f"Missing dependency: {req}"


def test_pyproject_dev_dependencies():
    """pyproject.toml lists all required dev dependencies."""
    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
    dep_names = [d.split(">=")[0].strip() for d in dev_deps]

    required = ["pytest", "pytest-asyncio", "pytest-cov", "ruff", "maturin"]
    for req in required:
        assert req in dep_names, f"Missing dev dependency: {req}"


def test_rust_crate_cargo_toml():
    """nemesis-parse/Cargo.toml exists and has correct package name."""
    import tomllib

    cargo_path = Path(__file__).parent.parent / "nemesis-parse" / "Cargo.toml"
    assert cargo_path.exists(), "nemesis-parse/Cargo.toml not found"

    with open(cargo_path, "rb") as f:
        cargo = tomllib.load(f)

    assert cargo["package"]["name"] == "nemesis-parse"
    assert "pyo3" in str(cargo["dependencies"])
    assert "tree-sitter" in str(cargo["dependencies"])


def test_gitignore_exists():
    """.gitignore exists and covers Python + Rust."""
    gitignore_path = Path(__file__).parent.parent / ".gitignore"
    assert gitignore_path.exists()

    content = gitignore_path.read_text()
    assert "__pycache__" in content
    assert "target/" in content
    assert ".venv" in content or "venv/" in content
    assert ".env" in content
```

**Step 2: Run test to verify it fails**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_integration_setup.py -v`
Expected: FAIL (da die Datei noch nicht existiert — erstmal anlegen, dann sollten alle Tests PASS sein wenn alles vorher korrekt aufgesetzt wurde)

**Step 3: Run test to verify it passes**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_integration_setup.py -v`
Expected: PASS (8 passed)

**Step 4: Gesamter Test-Lauf**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/ -v --tb=short`
Expected: PASS (alle Tests bestehen — ca. 29 Tests)

**Step 5: Commit**
```bash
git add tests/test_integration_setup.py
git commit -m "test: add integration test verifying complete project setup"
```

---

## Zusammenfassung

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 4 | Pydantic Config | `nemesis/core/config.py` | 11 Tests |
| 5 | CLI Skeleton | `nemesis/core/cli.py` | 6 Tests |
| 7 | Test-Infrastruktur | `tests/conftest.py` | 4 Tests |
| 10 | Integrationstest | `tests/test_integration_setup.py` | 8 Tests |

**Gesamt: 4 Tasks, ~29 Tests, 4 Commits**

---

**Vorheriges Paket:** [01a — Setup Basics](01a-setup-basics.md)
**Nächstes Paket:** [01c — Rust & DevOps](01c-rust-devops.md)
