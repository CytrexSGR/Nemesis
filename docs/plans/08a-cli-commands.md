# CLI Commands: init + index/status + query/rule

> **Arbeitspaket H1** — Teil 1 von 3 des CLI & Hooks Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the core CLI commands — `nemesis init` for project scaffolding, `nemesis index` and `nemesis status` for indexing pipeline integration, and `nemesis query` / `nemesis rule` for querying and rule management.

**Architecture:** The CLI (`nemesis/core/cli.py`) is the user-facing entry point built on Click. It delegates to the indexing pipeline, MCP server, and graph/vector stores.

**Tech Stack:** Python 3.11+, Click, pytest

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [06-mcp-server](06-mcp-server.md), [05-indexing-pipeline](05-indexing-pipeline.md)

**Tasks in this package:** 1, 2, 3 (von 10)

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

## Zusammenfassung (Arbeitspaket H1)

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 1 | `nemesis init` — Projekt-Initialisierung | `cli.py` | 5 |
| 2 | `nemesis index` und `nemesis status` | `cli.py` | 6 |
| 3 | `nemesis query` und `nemesis rule add/list` | `cli.py` | 7 |
| **Gesamt H1** | | **1 Impl-Datei, 1 Test-Datei** | **~18 Tests** |

---

**Navigation:**
- Vorheriges Paket: —
- Naechstes Paket: [08b-file-watcher.md](08b-file-watcher.md) — File Watcher + CLI watch (H2)
- Gesamtplan: [08-cli-hooks.md](08-cli-hooks.md)
