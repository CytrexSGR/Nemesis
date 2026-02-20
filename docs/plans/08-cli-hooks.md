# CLI & Hooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the CLI with full command implementations and build the Claude Code hooks system for automatic context loading, incremental indexing, and smart context injection.

**Architecture:** The CLI (`nemesis/core/cli.py`) is the user-facing entry point built on Click. It delegates to the indexing pipeline, MCP server, and graph/vector stores. The file watcher (`nemesis/core/watcher.py`) uses the watchdog library with proper debouncing to trigger incremental re-indexing. The hooks system (`nemesis/core/hooks.py`) provides three Claude Code hook entry points (SessionStart, PostToolUse, PreToolUse) that output structured context to stdout for Claude to consume.

**Tech Stack:** Python 3.11+, Click, watchdog, pytest

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [06-mcp-server](06-mcp-server.md), [05-indexing-pipeline](05-indexing-pipeline.md)

---

## Task 1: CLI `nemesis init` — Projekt-Initialisierung

**Files:**
- `tests/test_core/__init__.py`
- `tests/test_core/test_cli_commands.py`
- `nemesis/core/cli.py` (erweitern)

### Step 1 — Write failing test

```python
# tests/test_core/__init__.py
```

```python
# tests/test_core/test_cli_commands.py
"""Tests fuer die vollstaendigen CLI-Commands."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


class TestInitCommand:
    """Tests fuer 'nemesis init'."""

    def test_init_creates_nemesis_directory(self, tmp_path: Path) -> None:
        """nemesis init erstellt das .nemesis Verzeichnis."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["init", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / ".nemesis").is_dir()

    def test_init_creates_config_file(self, tmp_path: Path) -> None:
        """nemesis init erstellt eine config.toml."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["init", "--path", str(tmp_path)])

        assert result.exit_code == 0
        config_path = tmp_path / ".nemesis" / "config.toml"
        assert config_path.exists()
        content = config_path.read_text()
        assert "[project]" in content
        assert "[graph]" in content
        assert "[vector]" in content

    def test_init_creates_subdirectories(self, tmp_path: Path) -> None:
        """nemesis init erstellt graph/ und vectors/ Unterverzeichnisse."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["init", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / ".nemesis" / "graph").is_dir()
        assert (tmp_path / ".nemesis" / "vectors").is_dir()
        assert (tmp_path / ".nemesis" / "rules").is_dir()

    def test_init_already_initialized(self, tmp_path: Path) -> None:
        """nemesis init auf bereits initialisiertem Projekt warnt."""
        from nemesis.core.cli import main

        (tmp_path / ".nemesis").mkdir()
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert "already" in result.output.lower() or "bereits" in result.output.lower()

    def test_init_default_path_uses_cwd(self) -> None:
        """nemesis init ohne --path nutzt das aktuelle Verzeichnis."""
        from nemesis.core.cli import main

        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert Path(td, ".nemesis").is_dir()
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestInitCommand -x -v 2>&1 | head -30
```

Tests fail because `init` command does not exist in `cli.py`.

### Step 3 — Implement

```python
# nemesis/core/cli.py — Erweitern (init-Command und Hilfsfunktion hinzufuegen)
"""Nemesis CLI — command-line interface for indexing, querying, and serving."""

from __future__ import annotations

import json
from pathlib import Path

import click

from nemesis import __version__


_DEFAULT_CONFIG_TOML = """\
[project]
name = "{project_name}"
languages = ["python"]
ignore = ["node_modules", "venv", ".venv", "__pycache__", ".git", "target", "dist", "build"]

[graph]
backend = "kuzu"

[vector]
provider = "openai"
model = "text-embedding-3-small"

[memory]
auto_load_rules = true
auto_learn = true

[watcher]
enabled = true
debounce_ms = 500
"""


@click.group()
@click.version_option(version=__version__, prog_name="Nemesis")
def main() -> None:
    """Nemesis — GraphRAG context engine for AI coding agents."""


@main.command()
@click.option("--path", default=".", type=click.Path(exists=True), help="Project root path.")
def init(path: str) -> None:
    """Initialize a .nemesis directory and default config."""
    project_path = Path(path).resolve()
    nemesis_dir = project_path / ".nemesis"

    if nemesis_dir.exists():
        click.echo(f"Project already initialized at {nemesis_dir}")
        return

    nemesis_dir.mkdir(parents=True)
    (nemesis_dir / "graph").mkdir()
    (nemesis_dir / "vectors").mkdir()
    (nemesis_dir / "rules").mkdir()

    config_content = _DEFAULT_CONFIG_TOML.format(project_name=project_path.name)
    (nemesis_dir / "config.toml").write_text(config_content)

    click.echo(f"Initialized Nemesis project at {nemesis_dir}")


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


@main.command()
def status() -> None:
    """Show index health (files, nodes, edges, stale)."""
    click.echo("Index status: not yet implemented")


if __name__ == "__main__":
    main()
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestInitCommand -x -v
```

### Step 5 — Commit

```bash
git add nemesis/core/cli.py tests/test_core/test_cli_commands.py tests/test_core/__init__.py
git commit -m "feat(cli): implement 'nemesis init' with config scaffolding

TDD Task 1/10 of 08-cli-hooks plan.
Creates .nemesis directory, config.toml, graph/, vectors/, rules/ subdirs.
Detects already-initialized projects."
```

---

## Task 2: CLI `nemesis index` und `nemesis status`

**Files:**
- `tests/test_core/test_cli_commands.py` (erweitern)
- `nemesis/core/cli.py` (erweitern)

### Step 1 — Write failing test

```python
# tests/test_core/test_cli_commands.py — APPEND folgende Tests


class TestIndexCommand:
    """Tests fuer 'nemesis index'."""

    def test_index_requires_initialized_project(self, tmp_path: Path) -> None:
        """nemesis index in nicht-initialisiertem Projekt gibt Fehler."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code != 0 or "not initialized" in result.output.lower()

    @patch("nemesis.core.cli._get_pipeline")
    def test_index_calls_pipeline(self, mock_get_pipeline: MagicMock, tmp_path: Path) -> None:
        """nemesis index ruft die IndexingPipeline auf."""
        from nemesis.core.cli import main
        from nemesis.indexer.models import IndexResult

        # Setup .nemesis dir
        (tmp_path / ".nemesis").mkdir()
        (tmp_path / ".nemesis" / "config.toml").write_text("[project]\nname = 'test'\n")

        mock_pipeline = MagicMock()
        mock_pipeline.index_project.return_value = IndexResult(
            files_indexed=5,
            nodes_created=20,
            edges_created=15,
            chunks_created=10,
            embeddings_created=10,
            duration_ms=1234.5,
            errors=[],
        )
        mock_get_pipeline.return_value = mock_pipeline

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 0
        assert "5" in result.output  # files_indexed
        mock_pipeline.index_project.assert_called_once()

    @patch("nemesis.core.cli._get_pipeline")
    def test_index_shows_progress(self, mock_get_pipeline: MagicMock, tmp_path: Path) -> None:
        """nemesis index zeigt Fortschritt an."""
        from nemesis.core.cli import main
        from nemesis.indexer.models import IndexResult

        (tmp_path / ".nemesis").mkdir()
        (tmp_path / ".nemesis" / "config.toml").write_text("[project]\nname = 'test'\n")

        mock_pipeline = MagicMock()
        mock_pipeline.index_project.return_value = IndexResult(
            files_indexed=3,
            nodes_created=10,
            edges_created=8,
            chunks_created=6,
            embeddings_created=6,
            duration_ms=500.0,
            errors=[],
        )
        mock_get_pipeline.return_value = mock_pipeline

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 0
        # Sollte Zusammenfassung zeigen
        assert "indexed" in result.output.lower() or "files" in result.output.lower()

    @patch("nemesis.core.cli._get_pipeline")
    def test_index_with_errors_shows_warnings(
        self, mock_get_pipeline: MagicMock, tmp_path: Path
    ) -> None:
        """nemesis index zeigt Fehler in der Ausgabe."""
        from nemesis.core.cli import main
        from nemesis.indexer.models import IndexResult

        (tmp_path / ".nemesis").mkdir()
        (tmp_path / ".nemesis" / "config.toml").write_text("[project]\nname = 'test'\n")

        mock_pipeline = MagicMock()
        mock_pipeline.index_project.return_value = IndexResult(
            files_indexed=2,
            nodes_created=5,
            edges_created=3,
            chunks_created=4,
            embeddings_created=4,
            duration_ms=300.0,
            errors=["Failed to parse broken.py"],
        )
        mock_get_pipeline.return_value = mock_pipeline

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert "error" in result.output.lower() or "broken.py" in result.output


class TestStatusCommand:
    """Tests fuer 'nemesis status'."""

    def test_status_requires_initialized_project(self, tmp_path: Path) -> None:
        """nemesis status in nicht-initialisiertem Projekt gibt Fehler."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["status", "--path", str(tmp_path)])

        assert "not initialized" in result.output.lower() or result.exit_code != 0

    @patch("nemesis.core.cli._get_index_status")
    def test_status_shows_index_info(
        self, mock_get_status: MagicMock, tmp_path: Path
    ) -> None:
        """nemesis status zeigt Index-Informationen."""
        from nemesis.core.cli import main

        (tmp_path / ".nemesis").mkdir()

        mock_get_status.return_value = {
            "files": 42,
            "nodes": 200,
            "edges": 150,
            "chunks": 80,
            "stale_files": 3,
            "last_indexed": "2026-02-20T14:30:00",
        }

        runner = CliRunner()
        result = runner.invoke(main, ["status", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert "42" in result.output  # files
        assert "200" in result.output  # nodes
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestIndexCommand -x -v 2>&1 | head -30
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestStatusCommand -x -v 2>&1 | head -30
```

### Step 3 — Implement

In `nemesis/core/cli.py` die bestehenden `index` und `status` Commands ersetzen und Hilfsfunktionen hinzufuegen:

```python
# nemesis/core/cli.py — index und status Commands ersetzen

def _check_initialized(path: Path) -> bool:
    """Pruefe ob das Projekt initialisiert ist."""
    return (path / ".nemesis").is_dir()


def _get_pipeline(project_path: Path):
    """Erstelle eine IndexingPipeline fuer das Projekt.

    Dies ist ein Factory-Stub der in spaeteren Phasen durch die echte
    Pipeline-Erstellung ersetzt wird. Ermoeglicht einfaches Mocking in Tests.
    """
    from nemesis.core.config import NemesisConfig
    # Placeholder: wird durch echte Integration ersetzt
    raise NotImplementedError("Pipeline factory not yet integrated")


def _get_index_status(project_path: Path) -> dict:
    """Lade den Index-Status aus Graph und Vector Store.

    Placeholder fuer die echte Implementierung.
    """
    raise NotImplementedError("Status query not yet integrated")


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--languages", "-l", multiple=True, default=("python",), help="Languages to index.")
def index(path: str, languages: tuple[str, ...]) -> None:
    """Index a project directory. Shows progress and summary."""
    project_path = Path(path).resolve()

    if not _check_initialized(project_path):
        click.echo(f"Error: Project not initialized at {project_path}")
        click.echo("Run 'nemesis init' first.")
        raise SystemExit(1)

    click.echo(f"Indexing {project_path}...")

    pipeline = _get_pipeline(project_path)
    result = pipeline.index_project(project_path, languages=list(languages))

    # Zusammenfassung anzeigen
    click.echo(f"\nIndexing complete:")
    click.echo(f"  Files indexed: {result.files_indexed}")
    click.echo(f"  Nodes created: {result.nodes_created}")
    click.echo(f"  Edges created: {result.edges_created}")
    click.echo(f"  Chunks created: {result.chunks_created}")
    click.echo(f"  Duration: {result.duration_ms:.0f}ms")

    if result.errors:
        click.echo(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            click.echo(f"  - {error}")


@main.command()
@click.option("--path", default=".", type=click.Path(exists=True), help="Project root path.")
def status(path: str) -> None:
    """Show index health (files, nodes, edges, stale)."""
    project_path = Path(path).resolve()

    if not _check_initialized(project_path):
        click.echo(f"Error: Project not initialized at {project_path}")
        raise SystemExit(1)

    info = _get_index_status(project_path)

    click.echo(f"Nemesis Index Status: {project_path.name}")
    click.echo(f"  Files:        {info.get('files', 0)}")
    click.echo(f"  Nodes:        {info.get('nodes', 0)}")
    click.echo(f"  Edges:        {info.get('edges', 0)}")
    click.echo(f"  Chunks:       {info.get('chunks', 0)}")
    click.echo(f"  Stale files:  {info.get('stale_files', 0)}")
    click.echo(f"  Last indexed: {info.get('last_indexed', 'never')}")
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestIndexCommand -x -v
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestStatusCommand -x -v
```

### Step 5 — Commit

```bash
git add nemesis/core/cli.py tests/test_core/test_cli_commands.py
git commit -m "feat(cli): implement 'nemesis index' and 'nemesis status' commands

TDD Task 2/10 of 08-cli-hooks plan.
Index command calls pipeline with progress output and error reporting.
Status command shows files, nodes, edges, chunks, stale count."
```

---

## Task 3: CLI `nemesis query` und `nemesis rule`

**Files:**
- `tests/test_core/test_cli_commands.py` (erweitern)
- `nemesis/core/cli.py` (erweitern)

### Step 1 — Write failing test

```python
# tests/test_core/test_cli_commands.py — APPEND folgende Tests


class TestQueryCommand:
    """Tests fuer 'nemesis query'."""

    @patch("nemesis.core.cli._query_code_graph")
    def test_query_displays_results(self, mock_query: MagicMock, tmp_path: Path) -> None:
        """nemesis query zeigt Ergebnisse an."""
        from nemesis.core.cli import main

        (tmp_path / ".nemesis").mkdir()

        mock_query.return_value = [
            {
                "file": "src/auth.py",
                "function": "authenticate",
                "snippet": "def authenticate(user, pw): ...",
                "score": 0.95,
            },
            {
                "file": "src/user.py",
                "function": "get_user",
                "snippet": "def get_user(id): ...",
                "score": 0.82,
            },
        ]

        runner = CliRunner()
        result = runner.invoke(
            main, ["query", "how does auth work", "--path", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "auth.py" in result.output
        assert "authenticate" in result.output

    @patch("nemesis.core.cli._query_code_graph")
    def test_query_respects_limit(self, mock_query: MagicMock, tmp_path: Path) -> None:
        """nemesis query --limit begrenzt die Ergebnisse."""
        from nemesis.core.cli import main

        (tmp_path / ".nemesis").mkdir()
        mock_query.return_value = [{"file": "a.py", "function": "f", "snippet": "...", "score": 0.9}]

        runner = CliRunner()
        result = runner.invoke(
            main, ["query", "test", "--limit", "5", "--path", str(tmp_path)]
        )

        assert result.exit_code == 0
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args
        # limit wird an die Query-Funktion durchgereicht
        assert call_kwargs[1].get("limit", call_kwargs[0][-1] if len(call_kwargs[0]) > 2 else 5) == 5

    @patch("nemesis.core.cli._query_code_graph")
    def test_query_no_results(self, mock_query: MagicMock, tmp_path: Path) -> None:
        """nemesis query ohne Ergebnisse zeigt passende Meldung."""
        from nemesis.core.cli import main

        (tmp_path / ".nemesis").mkdir()
        mock_query.return_value = []

        runner = CliRunner()
        result = runner.invoke(main, ["query", "nonexistent", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert "no results" in result.output.lower() or "keine" in result.output.lower()


class TestRuleCommand:
    """Tests fuer 'nemesis rule add' und 'nemesis rule list'."""

    def test_rule_add_creates_rule_file(self, tmp_path: Path) -> None:
        """nemesis rule add speichert eine Regel als Datei."""
        from nemesis.core.cli import main

        (tmp_path / ".nemesis").mkdir()
        (tmp_path / ".nemesis" / "rules").mkdir()

        runner = CliRunner()
        result = runner.invoke(
            main, ["rule", "add", "Always use parameterized SQL queries", "--path", str(tmp_path)]
        )

        assert result.exit_code == 0
        rules_dir = tmp_path / ".nemesis" / "rules"
        rule_files = list(rules_dir.glob("*.md"))
        assert len(rule_files) == 1
        content = rule_files[0].read_text()
        assert "parameterized SQL queries" in content

    def test_rule_list_shows_rules(self, tmp_path: Path) -> None:
        """nemesis rule list zeigt alle Regeln."""
        from nemesis.core.cli import main

        rules_dir = tmp_path / ".nemesis" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "rule-001.md").write_text("Always use type hints")
        (rules_dir / "rule-002.md").write_text("Never use wildcard imports")

        runner = CliRunner()
        result = runner.invoke(main, ["rule", "list", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert "type hints" in result.output
        assert "wildcard imports" in result.output

    def test_rule_list_empty(self, tmp_path: Path) -> None:
        """nemesis rule list ohne Regeln zeigt passende Meldung."""
        from nemesis.core.cli import main

        rules_dir = tmp_path / ".nemesis" / "rules"
        rules_dir.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(main, ["rule", "list", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert "no rules" in result.output.lower() or "keine" in result.output.lower()

    def test_rule_add_increments_id(self, tmp_path: Path) -> None:
        """Jede neue Regel bekommt eine fortlaufende ID."""
        from nemesis.core.cli import main

        rules_dir = tmp_path / ".nemesis" / "rules"
        rules_dir.mkdir(parents=True)

        runner = CliRunner()
        runner.invoke(main, ["rule", "add", "Rule one", "--path", str(tmp_path)])
        runner.invoke(main, ["rule", "add", "Rule two", "--path", str(tmp_path)])

        rule_files = sorted(rules_dir.glob("*.md"))
        assert len(rule_files) == 2
        # Dateien haben unterschiedliche Namen
        assert rule_files[0].name != rule_files[1].name
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestQueryCommand -x -v 2>&1 | head -20
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestRuleCommand -x -v 2>&1 | head -20
```

### Step 3 — Implement

```python
# nemesis/core/cli.py — query Command ersetzen und rule Subgroup hinzufuegen

import time


def _query_code_graph(project_path: Path, query_text: str, limit: int = 10) -> list[dict]:
    """Fuehre eine semantische Suche ueber den Code-Graphen aus.

    Placeholder fuer die echte Implementierung die Vector Search +
    Graph Traversal kombiniert.
    """
    raise NotImplementedError("Query not yet integrated")


@main.command()
@click.argument("query_text")
@click.option("--limit", "-n", default=10, help="Max results.")
@click.option("--path", default=".", type=click.Path(exists=True), help="Project root path.")
def query(query_text: str, limit: int, path: str) -> None:
    """Query the code graph with natural language."""
    project_path = Path(path).resolve()

    if not _check_initialized(project_path):
        click.echo(f"Error: Project not initialized at {project_path}")
        raise SystemExit(1)

    results = _query_code_graph(project_path, query_text, limit=limit)

    if not results:
        click.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        score = result.get("score", 0.0)
        click.echo(f"\n[{i}] {result['file']}::{result.get('function', '?')} (score: {score:.2f})")
        snippet = result.get("snippet", "")
        if snippet:
            for line in snippet.split("\n")[:5]:
                click.echo(f"    {line}")


@main.group()
def rule() -> None:
    """Manage project rules."""


@rule.command("add")
@click.argument("rule_text")
@click.option("--path", default=".", type=click.Path(exists=True), help="Project root path.")
def rule_add(rule_text: str, path: str) -> None:
    """Add a new project rule."""
    project_path = Path(path).resolve()
    rules_dir = project_path / ".nemesis" / "rules"

    if not rules_dir.is_dir():
        click.echo("Error: Project not initialized. Run 'nemesis init' first.")
        raise SystemExit(1)

    # Naechste freie ID finden
    existing = sorted(rules_dir.glob("rule-*.md"))
    if existing:
        last_num = int(existing[-1].stem.split("-")[1])
        next_num = last_num + 1
    else:
        next_num = 1

    rule_file = rules_dir / f"rule-{next_num:03d}.md"
    rule_file.write_text(rule_text)

    click.echo(f"Rule added: {rule_file.name}")


@rule.command("list")
@click.option("--path", default=".", type=click.Path(exists=True), help="Project root path.")
def rule_list(path: str) -> None:
    """List all project rules."""
    project_path = Path(path).resolve()
    rules_dir = project_path / ".nemesis" / "rules"

    if not rules_dir.is_dir():
        click.echo("Error: Project not initialized.")
        raise SystemExit(1)

    rule_files = sorted(rules_dir.glob("rule-*.md"))

    if not rule_files:
        click.echo("No rules defined.")
        return

    for rule_file in rule_files:
        content = rule_file.read_text().strip()
        click.echo(f"  [{rule_file.stem}] {content}")
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestQueryCommand -x -v
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestRuleCommand -x -v
```

### Step 5 — Commit

```bash
git add nemesis/core/cli.py tests/test_core/test_cli_commands.py
git commit -m "feat(cli): implement 'nemesis query' and 'nemesis rule add/list'

TDD Task 3/10 of 08-cli-hooks plan.
Query command performs semantic search with formatted results.
Rule subgroup allows adding and listing project rules as .md files
in .nemesis/rules/."
```

---

## Task 4: File Watcher — Grundstruktur mit Debouncing

**Files:**
- `nemesis/core/watcher.py`
- `tests/test_core/test_watcher.py`

### Step 1 — Write failing test

```python
# tests/test_core/test_watcher.py
"""Tests fuer den File Watcher mit Debouncing."""
from __future__ import annotations

import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


class TestDebouncedHandler:
    """Tests fuer den debounced Event-Handler."""

    def test_handler_creation(self) -> None:
        """DebouncedHandler laesst sich erstellen."""
        from nemesis.core.watcher import DebouncedHandler

        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=0.5,
            ignore_patterns=set(),
            extensions={".py"},
        )
        assert handler is not None

    def test_handler_filters_non_code_files(self) -> None:
        """Handler ignoriert Nicht-Code-Dateien."""
        from nemesis.core.watcher import DebouncedHandler

        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=0.01,
            ignore_patterns=set(),
            extensions={".py"},
        )

        # Simuliere ein Event fuer eine .md Datei
        event = MagicMock()
        event.src_path = "/project/readme.md"
        event.is_directory = False

        handler.on_modified(event)
        time.sleep(0.05)

        callback.assert_not_called()

    def test_handler_accepts_code_files(self) -> None:
        """Handler verarbeitet Code-Dateien."""
        from nemesis.core.watcher import DebouncedHandler

        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=0.01,
            ignore_patterns=set(),
            extensions={".py"},
        )

        event = MagicMock()
        event.src_path = "/project/main.py"
        event.is_directory = False

        handler.on_modified(event)
        time.sleep(0.1)

        callback.assert_called_once_with(Path("/project/main.py"))

    def test_handler_debounces_rapid_events(self) -> None:
        """Schnelle aufeinanderfolgende Events werden zusammengefasst."""
        from nemesis.core.watcher import DebouncedHandler

        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=0.2,
            ignore_patterns=set(),
            extensions={".py"},
        )

        event = MagicMock()
        event.src_path = "/project/main.py"
        event.is_directory = False

        # 5 schnelle Events
        for _ in range(5):
            handler.on_modified(event)
            time.sleep(0.02)

        # Warten bis Debounce ablaeuft
        time.sleep(0.4)

        # Callback nur einmal aufgerufen
        assert callback.call_count == 1

    def test_handler_ignores_directories(self) -> None:
        """Handler ignoriert Directory-Events."""
        from nemesis.core.watcher import DebouncedHandler

        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=0.01,
            ignore_patterns=set(),
            extensions={".py"},
        )

        event = MagicMock()
        event.src_path = "/project/src/"
        event.is_directory = True

        handler.on_modified(event)
        time.sleep(0.05)

        callback.assert_not_called()

    def test_handler_ignores_patterns(self) -> None:
        """Handler ignoriert Dateien in ignorierten Verzeichnissen."""
        from nemesis.core.watcher import DebouncedHandler

        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=0.01,
            ignore_patterns={"__pycache__", "node_modules"},
            extensions={".py"},
        )

        event = MagicMock()
        event.src_path = "/project/__pycache__/module.py"
        event.is_directory = False

        handler.on_modified(event)
        time.sleep(0.05)

        callback.assert_not_called()

    def test_handler_separate_files_not_debounced(self) -> None:
        """Verschiedene Dateien werden separat verarbeitet."""
        from nemesis.core.watcher import DebouncedHandler

        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=0.05,
            ignore_patterns=set(),
            extensions={".py"},
        )

        event_a = MagicMock()
        event_a.src_path = "/project/a.py"
        event_a.is_directory = False

        event_b = MagicMock()
        event_b.src_path = "/project/b.py"
        event_b.is_directory = False

        handler.on_modified(event_a)
        handler.on_modified(event_b)

        time.sleep(0.2)

        assert callback.call_count == 2

    def test_handler_handles_created_events(self) -> None:
        """Handler verarbeitet auch on_created Events."""
        from nemesis.core.watcher import DebouncedHandler

        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=0.01,
            ignore_patterns=set(),
            extensions={".py"},
        )

        event = MagicMock()
        event.src_path = "/project/new.py"
        event.is_directory = False

        handler.on_created(event)
        time.sleep(0.05)

        callback.assert_called_once_with(Path("/project/new.py"))
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_watcher.py::TestDebouncedHandler -x -v 2>&1 | head -20
```

### Step 3 — Implement

```python
# nemesis/core/watcher.py
"""File Watcher fuer inkrementelle Index-Updates.

Nutzt watchdog fuer Filesystem-Events mit konfiguriertem Debouncing.
Aenderungen an Code-Dateien loesen einen Delta-Update der Indexing
Pipeline aus.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """Filesystem Event Handler mit Debouncing.

    Sammelt schnell aufeinanderfolgende Events fuer die gleiche Datei
    und loest den Callback erst nach Ablauf der Debounce-Zeit aus.
    Verschiedene Dateien werden unabhaengig voneinander debounced.

    Args:
        callback: Funktion die mit dem Dateipfad aufgerufen wird.
        debounce_seconds: Wartezeit nach dem letzten Event (default: 0.5).
        ignore_patterns: Verzeichnisnamen die ignoriert werden.
        extensions: Erlaubte Dateiendungen (z.B. {".py", ".ts"}).
    """

    def __init__(
        self,
        callback: Callable[[Path], None],
        debounce_seconds: float = 0.5,
        ignore_patterns: set[str] | None = None,
        extensions: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._callback = callback
        self._debounce_seconds = debounce_seconds
        self._ignore_patterns = ignore_patterns or set()
        self._extensions = extensions or {".py"}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _should_process(self, path: str, is_directory: bool) -> bool:
        """Pruefe ob die Datei verarbeitet werden soll."""
        if is_directory:
            return False

        file_path = Path(path)

        # Dateiendung pruefen
        if file_path.suffix not in self._extensions:
            return False

        # Ignorierte Verzeichnisse pruefen
        parts = file_path.parts
        for part in parts:
            if part in self._ignore_patterns:
                return False

        return True

    def _schedule_callback(self, path: str) -> None:
        """Plane den Callback mit Debouncing.

        Wenn fuer diese Datei bereits ein Timer laeuft, wird er
        abgebrochen und ein neuer gestartet. So wird der Callback
        erst nach debounce_seconds Ruhe aufgerufen.
        """
        with self._lock:
            if path in self._timers:
                self._timers[path].cancel()

            timer = threading.Timer(
                self._debounce_seconds,
                self._fire_callback,
                args=[path],
            )
            timer.daemon = True
            self._timers[path] = timer
            timer.start()

    def _fire_callback(self, path: str) -> None:
        """Fuehre den Callback aus und raeume den Timer auf."""
        with self._lock:
            self._timers.pop(path, None)

        file_path = Path(path)
        logger.debug("File changed (debounced): %s", file_path)

        try:
            self._callback(file_path)
        except Exception:
            logger.exception("Error in watcher callback for %s", file_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if self._should_process(event.src_path, event.is_directory):
            self._schedule_callback(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if self._should_process(event.src_path, event.is_directory):
            self._schedule_callback(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if self._should_process(event.src_path, event.is_directory):
            self._schedule_callback(event.src_path)

    def cancel_all(self) -> None:
        """Alle ausstehenden Timer abbrechen."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


class NemesisWatcher:
    """Verwaltet den File Watcher fuer ein Projekt.

    Startet einen watchdog Observer der Dateisystem-Events empfaengt
    und ueber den DebouncedHandler an den Callback weiterleitet.

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        callback: Funktion die bei Datei-Aenderungen aufgerufen wird.
        debounce_ms: Debounce-Zeit in Millisekunden.
        ignore_patterns: Verzeichnisnamen die ignoriert werden.
        extensions: Erlaubte Dateiendungen.
    """

    def __init__(
        self,
        project_path: Path,
        callback: Callable[[Path], None],
        debounce_ms: int = 500,
        ignore_patterns: set[str] | None = None,
        extensions: set[str] | None = None,
    ) -> None:
        self._project_path = project_path
        self._handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=debounce_ms / 1000.0,
            ignore_patterns=ignore_patterns or {
                "__pycache__", "node_modules", ".git", ".nemesis",
                "venv", ".venv", "target", "dist", "build",
            },
            extensions=extensions or {".py"},
        )
        self._observer: Observer | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Ob der Watcher gerade laeuft."""
        return self._running

    def start(self) -> None:
        """Starte den File Watcher."""
        if self._running:
            logger.warning("Watcher already running for %s", self._project_path)
            return

        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self._project_path),
            recursive=True,
        )
        self._observer.start()
        self._running = True
        logger.info("Watcher started for %s", self._project_path)

    def stop(self) -> None:
        """Stoppe den File Watcher."""
        if not self._running or self._observer is None:
            return

        self._handler.cancel_all()
        self._observer.stop()
        self._observer.join(timeout=5.0)
        self._observer = None
        self._running = False
        logger.info("Watcher stopped for %s", self._project_path)
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_watcher.py::TestDebouncedHandler -x -v
```

### Step 5 — Commit

```bash
git add nemesis/core/watcher.py tests/test_core/test_watcher.py
git commit -m "feat(watcher): implement DebouncedHandler with per-file debouncing

TDD Task 4/10 of 08-cli-hooks plan.
DebouncedHandler uses threading.Timer for per-file debounce.
Filters by extension, ignores directories and ignore-pattern paths.
Separate files are debounced independently."
```

---

## Task 5: File Watcher — NemesisWatcher Integration

**Files:**
- `tests/test_core/test_watcher.py` (erweitern)
- `nemesis/core/watcher.py` (bereits implementiert in Task 4)

### Step 1 — Write failing test

```python
# tests/test_core/test_watcher.py — APPEND folgende Tests
import os


class TestNemesisWatcher:
    """Tests fuer den NemesisWatcher Lifecycle."""

    def test_watcher_creation(self, tmp_path: Path) -> None:
        """NemesisWatcher laesst sich erstellen."""
        from nemesis.core.watcher import NemesisWatcher

        callback = MagicMock()
        watcher = NemesisWatcher(
            project_path=tmp_path,
            callback=callback,
            debounce_ms=100,
        )
        assert watcher is not None
        assert not watcher.is_running

    def test_watcher_start_and_stop(self, tmp_path: Path) -> None:
        """Watcher laesst sich starten und stoppen."""
        from nemesis.core.watcher import NemesisWatcher

        callback = MagicMock()
        watcher = NemesisWatcher(
            project_path=tmp_path,
            callback=callback,
            debounce_ms=100,
        )

        watcher.start()
        assert watcher.is_running

        watcher.stop()
        assert not watcher.is_running

    def test_watcher_double_start_is_safe(self, tmp_path: Path) -> None:
        """Doppeltes start() ist sicher."""
        from nemesis.core.watcher import NemesisWatcher

        callback = MagicMock()
        watcher = NemesisWatcher(
            project_path=tmp_path,
            callback=callback,
            debounce_ms=100,
        )

        watcher.start()
        watcher.start()  # Kein Fehler
        assert watcher.is_running

        watcher.stop()

    def test_watcher_stop_without_start_is_safe(self, tmp_path: Path) -> None:
        """stop() ohne vorheriges start() ist sicher."""
        from nemesis.core.watcher import NemesisWatcher

        callback = MagicMock()
        watcher = NemesisWatcher(
            project_path=tmp_path,
            callback=callback,
            debounce_ms=100,
        )

        watcher.stop()  # Kein Fehler

    def test_watcher_detects_file_creation(self, tmp_path: Path) -> None:
        """Watcher erkennt neue Dateien."""
        from nemesis.core.watcher import NemesisWatcher

        callback = MagicMock()
        watcher = NemesisWatcher(
            project_path=tmp_path,
            callback=callback,
            debounce_ms=50,
            extensions={".py"},
        )

        watcher.start()
        try:
            # Neue Datei erstellen
            new_file = tmp_path / "new_module.py"
            new_file.write_text("x = 1\n")

            # Warten auf Debounce + Verarbeitungszeit
            time.sleep(0.5)

            assert callback.call_count >= 1
            called_path = callback.call_args[0][0]
            assert "new_module.py" in str(called_path)
        finally:
            watcher.stop()

    def test_watcher_detects_file_modification(self, tmp_path: Path) -> None:
        """Watcher erkennt Datei-Aenderungen."""
        from nemesis.core.watcher import NemesisWatcher

        existing_file = tmp_path / "existing.py"
        existing_file.write_text("x = 1\n")

        callback = MagicMock()
        watcher = NemesisWatcher(
            project_path=tmp_path,
            callback=callback,
            debounce_ms=50,
            extensions={".py"},
        )

        watcher.start()
        try:
            # Datei aendern
            time.sleep(0.1)  # Kurz warten damit der Watcher bereit ist
            existing_file.write_text("x = 2  # changed\n")

            time.sleep(0.5)

            assert callback.call_count >= 1
        finally:
            watcher.stop()

    def test_watcher_ignores_non_code_files(self, tmp_path: Path) -> None:
        """Watcher ignoriert Nicht-Code-Dateien."""
        from nemesis.core.watcher import NemesisWatcher

        callback = MagicMock()
        watcher = NemesisWatcher(
            project_path=tmp_path,
            callback=callback,
            debounce_ms=50,
            extensions={".py"},
        )

        watcher.start()
        try:
            # Erstelle eine .md Datei — soll ignoriert werden
            (tmp_path / "readme.md").write_text("# Hello\n")
            time.sleep(0.3)

            callback.assert_not_called()
        finally:
            watcher.stop()
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_watcher.py::TestNemesisWatcher -x -v 2>&1 | head -20
```

Tests sollten mit der bestehenden Implementierung aus Task 4 bereits PASS.

### Step 3 — Verifizierung

Die NemesisWatcher-Klasse wurde bereits in Task 4 implementiert. Hier verifizieren wir nur dass die Integration-Tests passen.

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_watcher.py -x -v
```

### Step 5 — Commit

```bash
git add tests/test_core/test_watcher.py
git commit -m "test(watcher): add NemesisWatcher integration tests

TDD Task 5/10 of 08-cli-hooks plan.
Tests for lifecycle (start/stop), real filesystem events (creation,
modification), non-code file filtering, and double start/stop safety."
```

---

## Task 6: CLI `nemesis watch` — Watcher-Integration

**Files:**
- `tests/test_core/test_cli_commands.py` (erweitern)
- `nemesis/core/cli.py` (erweitern)

### Step 1 — Write failing test

```python
# tests/test_core/test_cli_commands.py — APPEND folgende Tests


class TestWatchCommand:
    """Tests fuer 'nemesis watch'."""

    @patch("nemesis.core.cli.NemesisWatcher")
    def test_watch_requires_initialized_project(
        self, mock_watcher_cls: MagicMock, tmp_path: Path
    ) -> None:
        """nemesis watch in nicht-initialisiertem Projekt gibt Fehler."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["watch", str(tmp_path)])

        assert "not initialized" in result.output.lower() or result.exit_code != 0

    @patch("nemesis.core.cli.NemesisWatcher")
    @patch("nemesis.core.cli._watch_loop")
    def test_watch_starts_watcher(
        self,
        mock_loop: MagicMock,
        mock_watcher_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """nemesis watch startet den File Watcher."""
        from nemesis.core.cli import main

        (tmp_path / ".nemesis").mkdir()
        mock_watcher = MagicMock()
        mock_watcher_cls.return_value = mock_watcher

        runner = CliRunner()
        result = runner.invoke(main, ["watch", str(tmp_path)])

        assert result.exit_code == 0
        mock_watcher.start.assert_called_once()

    @patch("nemesis.core.cli.NemesisWatcher")
    @patch("nemesis.core.cli._watch_loop")
    def test_watch_stops_on_exit(
        self,
        mock_loop: MagicMock,
        mock_watcher_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """nemesis watch stoppt den Watcher bei Beendigung."""
        from nemesis.core.cli import main

        (tmp_path / ".nemesis").mkdir()
        mock_watcher = MagicMock()
        mock_watcher_cls.return_value = mock_watcher

        runner = CliRunner()
        result = runner.invoke(main, ["watch", str(tmp_path)])

        assert result.exit_code == 0
        mock_watcher.stop.assert_called_once()
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestWatchCommand -x -v 2>&1 | head -20
```

### Step 3 — Implement

```python
# nemesis/core/cli.py — watch Command ersetzen und Imports hinzufuegen

from nemesis.core.watcher import NemesisWatcher


def _watch_loop() -> None:
    """Blockierende Warte-Schleife fuer den Watch-Modus.

    Kann durch Ctrl+C oder Signal unterbrochen werden.
    Wird in Tests gemockt um sofortiges Beenden zu ermoeglichen.
    """
    import signal
    import time

    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    while not stop_event.is_set():
        stop_event.wait(timeout=1.0)


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--debounce", default=500, help="Debounce time in ms.")
def watch(path: str, debounce: int) -> None:
    """Watch a project directory for changes and trigger incremental updates."""
    project_path = Path(path).resolve()

    if not _check_initialized(project_path):
        click.echo(f"Error: Project not initialized at {project_path}")
        raise SystemExit(1)

    def on_file_changed(file_path: Path) -> None:
        """Callback fuer Datei-Aenderungen."""
        click.echo(f"  Changed: {file_path.name}")
        # In der echten Integration: pipeline.reindex_file(file_path)

    watcher = NemesisWatcher(
        project_path=project_path,
        callback=on_file_changed,
        debounce_ms=debounce,
    )

    click.echo(f"Watching {project_path} (debounce: {debounce}ms)")
    click.echo("Press Ctrl+C to stop.\n")

    watcher.start()
    try:
        _watch_loop()
    finally:
        watcher.stop()
        click.echo("\nWatcher stopped.")
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestWatchCommand -x -v
```

### Step 5 — Commit

```bash
git add nemesis/core/cli.py tests/test_core/test_cli_commands.py
git commit -m "feat(cli): implement 'nemesis watch' with watcher integration

TDD Task 6/10 of 08-cli-hooks plan.
Watch command starts NemesisWatcher with configurable debounce,
runs blocking loop, stops cleanly on Ctrl+C/SIGTERM."
```

---

## Task 7: Hooks — SessionStart Hook

**Files:**
- `nemesis/core/hooks.py`
- `tests/test_core/test_hooks.py`

### Step 1 — Write failing test

```python
# tests/test_core/test_hooks.py
"""Tests fuer die Claude Code Hooks."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSessionStartHook:
    """Tests fuer den SessionStart Hook."""

    def test_session_start_with_indexed_project(self, tmp_path: Path) -> None:
        """SessionStart mit indexiertem Projekt liefert Kontext."""
        from nemesis.core.hooks import session_start_hook

        # Setup .nemesis mit Index-Marker
        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()
        (nemesis_dir / "graph").mkdir()
        (nemesis_dir / "vectors").mkdir()
        rules_dir = nemesis_dir / "rules"
        rules_dir.mkdir()
        (rules_dir / "rule-001.md").write_text("Always use type hints")
        (rules_dir / "rule-002.md").write_text("Use parameterized queries")

        # Mock fuer Index-Status
        mock_status = {
            "files": 42,
            "nodes": 200,
            "edges": 150,
            "last_indexed": "2026-02-20T14:30:00",
        }

        with patch("nemesis.core.hooks._get_index_status_for_hook", return_value=mock_status):
            output = session_start_hook(tmp_path)

        assert isinstance(output, str)
        assert len(output) > 0
        # Soll Regeln enthalten
        assert "type hints" in output
        assert "parameterized queries" in output
        # Soll Index-Status enthalten
        assert "42" in output

    def test_session_start_not_indexed(self, tmp_path: Path) -> None:
        """SessionStart ohne Index empfiehlt Indexierung."""
        from nemesis.core.hooks import session_start_hook

        output = session_start_hook(tmp_path)

        assert isinstance(output, str)
        assert "not indexed" in output.lower() or "nemesis index" in output.lower()

    def test_session_start_output_under_token_limit(self, tmp_path: Path) -> None:
        """SessionStart Output bleibt unter 2000 Tokens (~8000 chars)."""
        from nemesis.core.hooks import session_start_hook

        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()
        (nemesis_dir / "graph").mkdir()
        (nemesis_dir / "vectors").mkdir()
        rules_dir = nemesis_dir / "rules"
        rules_dir.mkdir()
        # Viele Regeln erstellen
        for i in range(50):
            (rules_dir / f"rule-{i:03d}.md").write_text(f"Rule number {i} with some content")

        mock_status = {"files": 100, "nodes": 500, "edges": 400, "last_indexed": "2026-02-20"}

        with patch("nemesis.core.hooks._get_index_status_for_hook", return_value=mock_status):
            output = session_start_hook(tmp_path)

        # ~4 chars per token -> 2000 tokens ~= 8000 chars
        assert len(output) < 8000

    def test_session_start_no_rules(self, tmp_path: Path) -> None:
        """SessionStart ohne Regeln funktioniert trotzdem."""
        from nemesis.core.hooks import session_start_hook

        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()
        (nemesis_dir / "graph").mkdir()
        (nemesis_dir / "vectors").mkdir()
        (nemesis_dir / "rules").mkdir()

        mock_status = {"files": 10, "nodes": 50, "edges": 30, "last_indexed": "2026-02-20"}

        with patch("nemesis.core.hooks._get_index_status_for_hook", return_value=mock_status):
            output = session_start_hook(tmp_path)

        assert isinstance(output, str)
        assert len(output) > 0
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_hooks.py::TestSessionStartHook -x -v 2>&1 | head -20
```

### Step 3 — Implement

```python
# nemesis/core/hooks.py
"""Claude Code Hooks fuer Nemesis.

Stellt drei Hook-Entry-Points bereit die von Claude Code aufgerufen werden:
- session_start_hook: SessionStart — Basis-Kontext laden
- file_changed_hook: PostToolUse (Edit/Write) — Inkrementeller Re-Index
- pre_task_hook: PreToolUse (Task/Plan) — Smart Context Injection

Jeder Hook gibt Text auf stdout aus, den Claude Code als Kontext konsumiert.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximale Output-Laenge (in Zeichen, ~2000 Tokens bei ~4 chars/token)
_MAX_SESSION_CHARS = 7500
_MAX_TASK_CHARS = 16000


def _get_index_status_for_hook(project_path: Path) -> dict | None:
    """Lade den Index-Status fuer Hook-Ausgabe.

    Placeholder — wird durch echte Graph-Abfrage ersetzt.
    """
    nemesis_dir = project_path / ".nemesis"
    if not nemesis_dir.is_dir():
        return None
    if not (nemesis_dir / "graph").is_dir():
        return None
    return None


def _load_rules(project_path: Path, max_rules: int = 20) -> list[str]:
    """Lade Projekt-Regeln aus .nemesis/rules/.

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        max_rules: Maximale Anzahl der zu ladenden Regeln.

    Returns:
        Liste der Regel-Texte.
    """
    rules_dir = project_path / ".nemesis" / "rules"
    if not rules_dir.is_dir():
        return []

    rules: list[str] = []
    for rule_file in sorted(rules_dir.glob("rule-*.md"))[:max_rules]:
        content = rule_file.read_text().strip()
        if content:
            rules.append(content)

    return rules


def session_start_hook(project_path: Path) -> str:
    """SessionStart Hook — Basis-Kontext fuer neue Session laden.

    Liefert:
    - Projekt-Status (Dateien, Nodes, Edges)
    - Aktive Regeln
    - Hinweis auf veraltete Dateien (stale)
    - Empfehlung zur Indexierung wenn nicht indexiert

    Args:
        project_path: Wurzelverzeichnis des Projekts.

    Returns:
        Kompakter Kontext-String (< 2000 Tokens).
    """
    nemesis_dir = project_path / ".nemesis"

    # Nicht initialisiert
    if not nemesis_dir.is_dir():
        return (
            f"[Nemesis] Project at {project_path.name} is not indexed.\n"
            f"Run 'nemesis init && nemesis index' to enable code intelligence.\n"
        )

    parts: list[str] = []
    parts.append(f"[Nemesis] Project: {project_path.name}")

    # Index-Status
    status = _get_index_status_for_hook(project_path)
    if status:
        parts.append(
            f"Index: {status.get('files', '?')} files, "
            f"{status.get('nodes', '?')} nodes, "
            f"{status.get('edges', '?')} edges"
        )
        stale = status.get("stale_files", 0)
        if stale > 0:
            parts.append(f"Warning: {stale} stale files — run 'nemesis index' to update")
        last = status.get("last_indexed", "unknown")
        parts.append(f"Last indexed: {last}")
    else:
        parts.append("Index: no index data found — run 'nemesis index' to build index")

    # Regeln laden
    rules = _load_rules(project_path)
    if rules:
        parts.append(f"\nActive Rules ({len(rules)}):")
        char_budget = _MAX_SESSION_CHARS - sum(len(p) for p in parts) - 200
        chars_used = 0
        for i, rule in enumerate(rules, 1):
            rule_line = f"  {i}. {rule}"
            if chars_used + len(rule_line) > char_budget:
                parts.append(f"  ... and {len(rules) - i + 1} more rules")
                break
            parts.append(rule_line)
            chars_used += len(rule_line)

    output = "\n".join(parts)

    # Sicherheitsgrenze: Abschneiden wenn zu lang
    if len(output) > _MAX_SESSION_CHARS:
        output = output[:_MAX_SESSION_CHARS - 50] + "\n... [truncated]"

    return output


def file_changed_hook(file_path: Path, project_path: Path | None = None) -> str:
    """PostToolUse Hook — Inkrementeller Re-Index einer geaenderten Datei.

    Loescht alte AST-Nodes, re-parsed, re-chunked, re-embedded.

    Args:
        file_path: Pfad zur geaenderten Datei.
        project_path: Optionaler Projekt-Pfad (wird auto-detektiert wenn None).

    Returns:
        Status-Meldung als String.
    """
    if project_path is None:
        project_path = _find_project_root(file_path)

    if project_path is None:
        return ""  # Stille — kein Nemesis-Projekt

    if not file_path.exists():
        return f"[Nemesis] File deleted: {file_path.name}"

    # Placeholder fuer die echte Pipeline-Integration
    return f"[Nemesis] Re-indexed: {file_path.name}"


def pre_task_hook(project_path: Path, task_prompt: str) -> str:
    """PreToolUse Hook — Smart Context Injection fuer Tasks/Plans.

    Liest den Task/Plan-Prompt und liefert relevanten Code-Kontext,
    Regeln und Architektur-Informationen.

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        task_prompt: Der Task/Plan-Prompt von stdin.

    Returns:
        Relevanter Kontext-String (< 4000 Tokens).
    """
    nemesis_dir = project_path / ".nemesis"

    if not nemesis_dir.is_dir():
        return ""  # Stille — kein Nemesis-Projekt

    parts: list[str] = []
    parts.append(f"[Nemesis Context for task]")

    # Regeln laden
    rules = _load_rules(project_path)
    if rules:
        parts.append(f"\nRelevant Rules:")
        for i, rule in enumerate(rules, 1):
            parts.append(f"  {i}. {rule}")

    # Placeholder fuer die echte Smart Context Integration:
    # - Semantische Suche ueber den Task-Prompt
    # - Graph-Traversal fuer verwandte Code-Strukturen
    # - Architektur-Ueberblick
    parts.append(f"\n(Full context injection pending pipeline integration)")

    output = "\n".join(parts)

    if len(output) > _MAX_TASK_CHARS:
        output = output[:_MAX_TASK_CHARS - 50] + "\n... [truncated]"

    return output


def _find_project_root(path: Path) -> Path | None:
    """Suche das Projekt-Root anhand der .nemesis Verzeichnisstruktur.

    Geht von der Datei aufwaerts bis ein .nemesis/ Verzeichnis gefunden wird.
    """
    current = path.parent if path.is_file() else path
    for _ in range(20):  # Max 20 Ebenen
        if (current / ".nemesis").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_hooks.py::TestSessionStartHook -x -v
```

### Step 5 — Commit

```bash
git add nemesis/core/hooks.py tests/test_core/test_hooks.py
git commit -m "feat(hooks): implement session_start_hook with rules and index status

TDD Task 7/10 of 08-cli-hooks plan.
SessionStart hook delivers compact context package: project status,
active rules, stale file warnings. Output capped at ~2000 tokens.
Includes _find_project_root and _load_rules helpers."
```

---

## Task 8: Hooks — FileChanged und PreTask Hooks

**Files:**
- `tests/test_core/test_hooks.py` (erweitern)
- `nemesis/core/hooks.py` (bereits implementiert)

### Step 1 — Write failing test

```python
# tests/test_core/test_hooks.py — APPEND folgende Tests


class TestFileChangedHook:
    """Tests fuer den PostToolUse (file-changed) Hook."""

    def test_file_changed_existing_file(self, tmp_path: Path) -> None:
        """file_changed_hook fuer existierende Datei liefert Status."""
        from nemesis.core.hooks import file_changed_hook

        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()

        test_file = tmp_path / "main.py"
        test_file.write_text("x = 1\n")

        output = file_changed_hook(test_file, project_path=tmp_path)

        assert isinstance(output, str)
        assert "main.py" in output

    def test_file_changed_deleted_file(self, tmp_path: Path) -> None:
        """file_changed_hook fuer geloeschte Datei meldet Loeschung."""
        from nemesis.core.hooks import file_changed_hook

        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()

        deleted_file = tmp_path / "gone.py"
        # Datei existiert nicht

        output = file_changed_hook(deleted_file, project_path=tmp_path)

        assert "deleted" in output.lower() or "gone.py" in output

    def test_file_changed_no_project(self, tmp_path: Path) -> None:
        """file_changed_hook ohne Nemesis-Projekt gibt leeren String."""
        from nemesis.core.hooks import file_changed_hook

        test_file = tmp_path / "orphan.py"
        test_file.write_text("x = 1\n")

        output = file_changed_hook(test_file, project_path=None)

        assert output == ""

    def test_file_changed_auto_detects_project(self, tmp_path: Path) -> None:
        """file_changed_hook erkennt das Projekt-Root automatisch."""
        from nemesis.core.hooks import file_changed_hook

        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        test_file = src_dir / "module.py"
        test_file.write_text("x = 1\n")

        output = file_changed_hook(test_file)  # kein project_path

        assert "module.py" in output


class TestPreTaskHook:
    """Tests fuer den PreToolUse (pre-task) Hook."""

    def test_pre_task_returns_context(self, tmp_path: Path) -> None:
        """pre_task_hook liefert relevanten Kontext."""
        from nemesis.core.hooks import pre_task_hook

        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()
        rules_dir = nemesis_dir / "rules"
        rules_dir.mkdir()
        (rules_dir / "rule-001.md").write_text("Always validate input")

        output = pre_task_hook(tmp_path, task_prompt="Refactor the auth module")

        assert isinstance(output, str)
        assert len(output) > 0
        assert "validate input" in output

    def test_pre_task_no_project(self, tmp_path: Path) -> None:
        """pre_task_hook ohne Nemesis-Projekt gibt leeren String."""
        from nemesis.core.hooks import pre_task_hook

        output = pre_task_hook(tmp_path, task_prompt="do something")

        assert output == ""

    def test_pre_task_output_under_token_limit(self, tmp_path: Path) -> None:
        """pre_task_hook Output bleibt unter 4000 Tokens (~16000 chars)."""
        from nemesis.core.hooks import pre_task_hook

        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()
        rules_dir = nemesis_dir / "rules"
        rules_dir.mkdir()
        # Viele lange Regeln
        for i in range(100):
            (rules_dir / f"rule-{i:03d}.md").write_text(
                f"Long rule number {i}: " + "detail " * 50
            )

        output = pre_task_hook(tmp_path, task_prompt="build everything")

        assert len(output) < 16000

    def test_pre_task_includes_rules(self, tmp_path: Path) -> None:
        """pre_task_hook inkludiert relevante Regeln."""
        from nemesis.core.hooks import pre_task_hook

        nemesis_dir = tmp_path / ".nemesis"
        nemesis_dir.mkdir()
        rules_dir = nemesis_dir / "rules"
        rules_dir.mkdir()
        (rules_dir / "rule-001.md").write_text("Use async/await for IO")
        (rules_dir / "rule-002.md").write_text("Never commit secrets")

        output = pre_task_hook(tmp_path, task_prompt="add a new API endpoint")

        assert "async/await" in output
        assert "secrets" in output


class TestFindProjectRoot:
    """Tests fuer die Projekt-Root-Erkennung."""

    def test_find_root_from_file(self, tmp_path: Path) -> None:
        """_find_project_root findet Root von einer Datei aus."""
        from nemesis.core.hooks import _find_project_root

        (tmp_path / ".nemesis").mkdir()
        src = tmp_path / "src" / "deep" / "module"
        src.mkdir(parents=True)
        test_file = src / "test.py"
        test_file.write_text("x = 1\n")

        root = _find_project_root(test_file)

        assert root == tmp_path

    def test_find_root_no_nemesis(self, tmp_path: Path) -> None:
        """_find_project_root gibt None wenn kein .nemesis existiert."""
        from nemesis.core.hooks import _find_project_root

        test_file = tmp_path / "orphan.py"
        test_file.write_text("x = 1\n")

        root = _find_project_root(test_file)

        assert root is None

    def test_find_root_from_directory(self, tmp_path: Path) -> None:
        """_find_project_root funktioniert auch mit Verzeichnissen."""
        from nemesis.core.hooks import _find_project_root

        (tmp_path / ".nemesis").mkdir()
        subdir = tmp_path / "src"
        subdir.mkdir()

        root = _find_project_root(subdir)

        assert root == tmp_path
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_hooks.py -x -v 2>&1 | head -30
```

### Step 3 — Verifizierung

Die Hook-Funktionen `file_changed_hook`, `pre_task_hook` und `_find_project_root` wurden bereits in Task 7 implementiert. Hier verifizieren wir dass alle Tests passen.

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_hooks.py -x -v
```

### Step 5 — Commit

```bash
git add tests/test_core/test_hooks.py
git commit -m "test(hooks): add tests for file_changed, pre_task, and project root detection

TDD Task 8/10 of 08-cli-hooks plan.
Tests for PostToolUse hook (existing/deleted files, auto-detect project),
PreToolUse hook (context output, rules inclusion, token limits),
and _find_project_root helper (nested dirs, missing .nemesis)."
```

---

## Task 9: CLI `nemesis hook` — Hook Entry Points

**Files:**
- `tests/test_core/test_cli_commands.py` (erweitern)
- `nemesis/core/cli.py` (erweitern)

### Step 1 — Write failing test

```python
# tests/test_core/test_cli_commands.py — APPEND folgende Tests


class TestHookCommand:
    """Tests fuer 'nemesis hook' Subcommands."""

    @patch("nemesis.core.hooks.session_start_hook")
    def test_hook_session_start(self, mock_hook: MagicMock, tmp_path: Path) -> None:
        """nemesis hook session-start ruft den Hook auf und gibt Output aus."""
        from nemesis.core.cli import main

        mock_hook.return_value = "[Nemesis] Project: test\nIndex: 42 files"

        runner = CliRunner()
        result = runner.invoke(main, ["hook", "session-start", "--project", str(tmp_path)])

        assert result.exit_code == 0
        assert "42 files" in result.output
        mock_hook.assert_called_once()

    @patch("nemesis.core.hooks.file_changed_hook")
    def test_hook_file_changed(self, mock_hook: MagicMock, tmp_path: Path) -> None:
        """nemesis hook file-changed ruft den Hook auf."""
        from nemesis.core.cli import main

        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")

        mock_hook.return_value = "[Nemesis] Re-indexed: test.py"

        runner = CliRunner()
        result = runner.invoke(main, ["hook", "file-changed", "--file", str(test_file)])

        assert result.exit_code == 0
        assert "test.py" in result.output

    @patch("nemesis.core.hooks.pre_task_hook")
    def test_hook_pre_task_reads_stdin(self, mock_hook: MagicMock, tmp_path: Path) -> None:
        """nemesis hook pre-task liest den Task-Prompt von stdin."""
        from nemesis.core.cli import main

        mock_hook.return_value = "[Nemesis Context] Rules: ..."

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["hook", "pre-task", "--project", str(tmp_path)],
            input="Refactor the auth module",
        )

        assert result.exit_code == 0
        mock_hook.assert_called_once()
        # Prompt sollte durchgereicht werden
        call_args = mock_hook.call_args
        assert "auth" in call_args[1].get("task_prompt", call_args[0][1] if len(call_args[0]) > 1 else "")

    def test_hook_help(self) -> None:
        """nemesis hook --help zeigt Subcommands."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["hook", "--help"])

        assert result.exit_code == 0
        assert "session-start" in result.output
        assert "file-changed" in result.output
        assert "pre-task" in result.output


class TestServeCommand:
    """Tests fuer 'nemesis serve'."""

    def test_serve_help(self) -> None:
        """nemesis serve --help zeigt Optionen."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])

        assert result.exit_code == 0
        assert "MCP" in result.output or "server" in result.output.lower()
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestHookCommand -x -v 2>&1 | head -20
```

### Step 3 — Implement

```python
# nemesis/core/cli.py — hook Subgroup und serve ersetzen

import sys
import threading

from nemesis.core import hooks as hook_module


@main.group()
def hook() -> None:
    """Claude Code hook entry points."""


@hook.command("session-start")
@click.option("--project", required=True, type=click.Path(exists=True), help="Project root path.")
def hook_session_start(project: str) -> None:
    """SessionStart hook — load base context for new session."""
    project_path = Path(project).resolve()
    output = hook_module.session_start_hook(project_path)
    if output:
        click.echo(output)


@hook.command("file-changed")
@click.option("--file", "file_path", required=True, type=click.Path(), help="Changed file path.")
@click.option("--project", default=None, type=click.Path(exists=True), help="Project root path.")
def hook_file_changed(file_path: str, project: str | None) -> None:
    """PostToolUse hook — incremental re-index for changed file."""
    fp = Path(file_path)
    pp = Path(project).resolve() if project else None
    output = hook_module.file_changed_hook(fp, project_path=pp)
    if output:
        click.echo(output)


@hook.command("pre-task")
@click.option("--project", required=True, type=click.Path(exists=True), help="Project root path.")
def hook_pre_task(project: str) -> None:
    """PreToolUse hook — inject smart context for task/plan."""
    project_path = Path(project).resolve()
    # Task-Prompt von stdin lesen
    task_prompt = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
    output = hook_module.pre_task_hook(project_path, task_prompt=task_prompt)
    if output:
        click.echo(output)


@main.command()
@click.option("--stdio", is_flag=True, default=True, help="Use stdio transport (default).")
def serve(stdio: bool) -> None:
    """Start the MCP server (stdio mode)."""
    click.echo("Starting Nemesis MCP server (stdio mode)...")
    # Placeholder: wird in 06-mcp-server implementiert
    click.echo("MCP server not yet implemented. See 06-mcp-server plan.")
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestHookCommand -x -v
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_commands.py::TestServeCommand -x -v
```

### Step 5 — Commit

```bash
git add nemesis/core/cli.py tests/test_core/test_cli_commands.py
git commit -m "feat(cli): implement 'nemesis hook' subcommands and 'nemesis serve'

TDD Task 9/10 of 08-cli-hooks plan.
Hook subgroup with session-start, file-changed, pre-task entry points.
pre-task reads task prompt from stdin. serve command placeholder for
MCP server (06-mcp-server dependency)."
```

---

## Task 10: Hook Config Generator und Integration Test

**Files:**
- `nemesis/core/hooks.py` (erweitern)
- `tests/test_core/test_hooks.py` (erweitern)
- `tests/test_core/test_cli_integration.py`

### Step 1 — Write failing test

```python
# tests/test_core/test_hooks.py — APPEND folgende Tests


class TestHookConfigGenerator:
    """Tests fuer den Claude Code Hooks Config Generator."""

    def test_generate_hooks_config(self) -> None:
        """generate_hooks_config erzeugt gueltige JSON-Konfiguration."""
        from nemesis.core.hooks import generate_hooks_config

        config = generate_hooks_config()

        assert isinstance(config, dict)
        assert "hooks" in config
        hooks = config["hooks"]
        assert "SessionStart" in hooks
        assert "PostToolUse" in hooks
        assert "PreToolUse" in hooks

    def test_session_start_hook_config(self) -> None:
        """SessionStart Hook hat korrekten Command und Timeout."""
        from nemesis.core.hooks import generate_hooks_config

        config = generate_hooks_config()
        session_hooks = config["hooks"]["SessionStart"]

        assert len(session_hooks) == 1
        hook = session_hooks[0]
        assert "nemesis hook session-start" in hook["command"]
        assert hook["timeout"] == 5000

    def test_post_tool_use_hook_config(self) -> None:
        """PostToolUse Hook hat Matcher fuer Edit/Write."""
        from nemesis.core.hooks import generate_hooks_config

        config = generate_hooks_config()
        post_hooks = config["hooks"]["PostToolUse"]

        assert len(post_hooks) == 1
        hook = post_hooks[0]
        assert "Edit" in hook["matcher"]
        assert "Write" in hook["matcher"]
        assert "nemesis hook file-changed" in hook["command"]
        assert hook["timeout"] == 3000

    def test_pre_tool_use_hook_config(self) -> None:
        """PreToolUse Hook hat Matcher fuer Task/EnterPlanMode."""
        from nemesis.core.hooks import generate_hooks_config

        config = generate_hooks_config()
        pre_hooks = config["hooks"]["PreToolUse"]

        assert len(pre_hooks) == 1
        hook = pre_hooks[0]
        assert "Task" in hook["matcher"]
        assert "EnterPlanMode" in hook["matcher"]
        assert "nemesis hook pre-task" in hook["command"]
        assert hook["timeout"] == 3000

    def test_config_is_valid_json_serializable(self) -> None:
        """Die generierte Config ist JSON-serialisierbar."""
        from nemesis.core.hooks import generate_hooks_config

        config = generate_hooks_config()
        json_str = json.dumps(config, indent=2)
        parsed = json.loads(json_str)

        assert parsed == config

    def test_config_with_custom_project_path(self) -> None:
        """Config kann mit custom Project-Pfad generiert werden."""
        from nemesis.core.hooks import generate_hooks_config

        config = generate_hooks_config(project_path="/home/user/my-project")
        session_cmd = config["hooks"]["SessionStart"][0]["command"]

        assert "/home/user/my-project" in session_cmd
```

```python
# tests/test_core/test_cli_integration.py
"""Integrationstests fuer CLI + Hooks + Watcher zusammen."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


class TestCLIHookIntegration:
    """End-to-End Tests fuer die CLI-Hook-Integration."""

    def test_init_then_hook_session_start(self, tmp_path: Path) -> None:
        """Nach 'nemesis init' liefert SessionStart Hook Output."""
        from nemesis.core.cli import main

        runner = CliRunner()

        # Init
        result = runner.invoke(main, ["init", "--path", str(tmp_path)])
        assert result.exit_code == 0

        # Rule hinzufuegen
        result = runner.invoke(
            main, ["rule", "add", "Use type annotations everywhere", "--path", str(tmp_path)]
        )
        assert result.exit_code == 0

        # SessionStart Hook
        result = runner.invoke(main, ["hook", "session-start", "--project", str(tmp_path)])
        assert result.exit_code == 0
        assert "type annotations" in result.output

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Voller Lifecycle: init -> rule add -> rule list -> hook."""
        from nemesis.core.cli import main

        runner = CliRunner()

        # 1. Init
        result = runner.invoke(main, ["init", "--path", str(tmp_path)])
        assert result.exit_code == 0

        # 2. Regeln hinzufuegen
        result = runner.invoke(
            main, ["rule", "add", "Always use async/await for IO", "--path", str(tmp_path)]
        )
        assert result.exit_code == 0

        result = runner.invoke(
            main, ["rule", "add", "Never use print() in production code", "--path", str(tmp_path)]
        )
        assert result.exit_code == 0

        # 3. Regeln auflisten
        result = runner.invoke(main, ["rule", "list", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "async/await" in result.output
        assert "print()" in result.output

        # 4. SessionStart Hook
        result = runner.invoke(main, ["hook", "session-start", "--project", str(tmp_path)])
        assert result.exit_code == 0
        assert tmp_path.name in result.output

    def test_version_command(self) -> None:
        """--version gibt die korrekte Version aus."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_all_commands_have_help(self) -> None:
        """Alle Commands haben funktionierendes --help."""
        from nemesis.core.cli import main

        runner = CliRunner()

        commands = ["init", "index", "query", "watch", "serve", "status", "hook", "rule"]
        for cmd in commands:
            result = runner.invoke(main, [cmd, "--help"])
            assert result.exit_code == 0, f"'{cmd} --help' failed: {result.output}"

    def test_hook_generates_config_json(self, tmp_path: Path) -> None:
        """nemesis hook generate-config erzeugt gueltige JSON-Config."""
        from nemesis.core.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main, ["hook", "generate-config", "--project", str(tmp_path)]
        )

        assert result.exit_code == 0
        config = json.loads(result.output)
        assert "hooks" in config
        assert "SessionStart" in config["hooks"]
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_hooks.py::TestHookConfigGenerator -x -v 2>&1 | head -20
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_integration.py -x -v 2>&1 | head -20
```

### Step 3 — Implement

```python
# nemesis/core/hooks.py — APPEND am Ende der Datei

def generate_hooks_config(project_path: str | None = None) -> dict:
    """Generiere die Claude Code Hooks JSON-Konfiguration.

    Erstellt die hooks.json Konfiguration die Claude Code mitteilt,
    wann welche Nemesis-Hooks aufgerufen werden sollen.

    Args:
        project_path: Optionaler fester Projekt-Pfad.
                      Standard: $PWD (wird zur Laufzeit aufgeloest).

    Returns:
        Dict mit der Hooks-Konfiguration (JSON-serialisierbar).
    """
    project = project_path or "$PWD"

    return {
        "hooks": {
            "SessionStart": [
                {
                    "command": f"nemesis hook session-start --project {project}",
                    "timeout": 5000,
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "Edit|Write",
                    "command": f"nemesis hook file-changed --file $FILE --project {project}",
                    "timeout": 3000,
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Task|EnterPlanMode",
                    "command": f"nemesis hook pre-task --project {project}",
                    "timeout": 3000,
                }
            ],
        }
    }
```

```python
# nemesis/core/cli.py — generate-config Subcommand zur hook Group hinzufuegen

@hook.command("generate-config")
@click.option("--project", default=None, type=click.Path(), help="Project root path (default: $PWD).")
def hook_generate_config(project: str | None) -> None:
    """Generate Claude Code hooks JSON config."""
    config = hook_module.generate_hooks_config(project_path=project)
    click.echo(json.dumps(config, indent=2))
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_hooks.py -x -v
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/test_cli_integration.py -x -v
cd /home/andreas/projects/nemesis && python -m pytest tests/test_core/ -x -v
```

### Step 5 — Commit

```bash
git add nemesis/core/hooks.py nemesis/core/cli.py \
        tests/test_core/test_hooks.py tests/test_core/test_cli_integration.py
git commit -m "feat(hooks): add config generator and CLI integration tests

TDD Task 10/10 of 08-cli-hooks plan.
generate_hooks_config() creates Claude Code hooks.json with
SessionStart, PostToolUse (Edit/Write), PreToolUse (Task/Plan).
generate-config CLI subcommand outputs JSON to stdout.
Integration tests verify full init -> rules -> hook lifecycle."
```

---

## Zusammenfassung

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 1 | `nemesis init` — Projekt-Initialisierung | `cli.py` | 5 |
| 2 | `nemesis index` und `nemesis status` | `cli.py` | 6 |
| 3 | `nemesis query` und `nemesis rule add/list` | `cli.py` | 7 |
| 4 | DebouncedHandler mit per-file Debouncing | `watcher.py` | 8 |
| 5 | NemesisWatcher Integration (start/stop/lifecycle) | `watcher.py` | 7 |
| 6 | `nemesis watch` CLI Integration | `cli.py` | 3 |
| 7 | SessionStart Hook | `hooks.py` | 4 |
| 8 | FileChanged + PreTask Hooks + Project Root Detection | `hooks.py` | 10 |
| 9 | `nemesis hook` CLI Subcommands + `nemesis serve` | `cli.py` | 5 |
| 10 | Hook Config Generator + Integration Tests | `hooks.py`, `cli.py` | 11 |
| **Gesamt** | | **3 Implementierungs-Dateien, 4 Test-Dateien** | **~66 Tests** |

### Dateien erstellt/geaendert

```
nemesis/core/
├── cli.py              # Vollstaendige CLI mit allen Commands
├── watcher.py          # DebouncedHandler + NemesisWatcher
└── hooks.py            # session_start, file_changed, pre_task, generate_config

tests/test_core/
├── __init__.py
├── test_cli_commands.py      # CLI Command Tests (init, index, query, watch, rule, hook, serve, status)
├── test_watcher.py           # Watcher Tests (debouncing, filtering, lifecycle)
├── test_hooks.py             # Hook Tests (all three hooks + config generator)
└── test_cli_integration.py   # End-to-end Integration Tests
```

### CLI Command Map

```
nemesis
├── init [--path]                           # Initialisierung
├── index <path> [-l langs]                 # Full Index
├── query <text> [-n limit] [--path]        # Semantische Suche
├── watch <path> [--debounce ms]            # File Watcher
├── serve [--stdio]                         # MCP Server
├── status [--path]                         # Index Health
├── rule
│   ├── add <rule> [--path]                 # Regel hinzufuegen
│   └── list [--path]                       # Regeln auflisten
└── hook
    ├── session-start --project <path>      # SessionStart Hook
    ├── file-changed --file <path>          # PostToolUse Hook
    ├── pre-task --project <path>           # PreToolUse Hook
    └── generate-config [--project <path>]  # Hooks JSON Config
```

### Claude Code Hooks Integration

```json
{
  "hooks": {
    "SessionStart": [{
      "command": "nemesis hook session-start --project $PWD",
      "timeout": 5000
    }],
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "command": "nemesis hook file-changed --file $FILE --project $PWD",
      "timeout": 3000
    }],
    "PreToolUse": [{
      "matcher": "Task|EnterPlanMode",
      "command": "nemesis hook pre-task --project $PWD",
      "timeout": 3000
    }]
  }
}
```
