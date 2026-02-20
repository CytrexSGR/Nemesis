# Hooks + Config Generator

> **Arbeitspaket H3** — Teil 3 von 3 des CLI & Hooks Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Claude Code hooks system (`session_start_hook`, `file_changed_hook`, `pre_task_hook`), the hooks config generator, the CLI `nemesis hook` subcommands, and end-to-end integration tests for the full init-rules-hook lifecycle.

**Architecture:** The hooks system (`nemesis/core/hooks.py`) provides three Claude Code hook entry points (SessionStart, PostToolUse, PreToolUse) that output structured context to stdout for Claude to consume. The CLI `nemesis hook` subgroup exposes these as CLI commands. The config generator creates the `hooks.json` for Claude Code integration.

**Tech Stack:** Python 3.11+, Click, pytest

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [08a-cli-commands.md](08a-cli-commands.md) (Task 1-3 fuer CLI-Grundstruktur, `_check_initialized`, `rule` Subgroup)

**Tasks in this package:** 7, 8, 9, 10 (von 10)

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

## Zusammenfassung (Arbeitspaket H3)

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 7 | SessionStart Hook | `hooks.py` | 4 |
| 8 | FileChanged + PreTask Hooks + Project Root Detection | `hooks.py` | 10 |
| 9 | `nemesis hook` CLI Subcommands + `nemesis serve` | `cli.py` | 5 |
| 10 | Hook Config Generator + Integration Tests | `hooks.py`, `cli.py` | 11 |
| **Gesamt H3** | | **2 Impl-Dateien, 3 Test-Dateien** | **~30 Tests** |

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

---

**Navigation:**
- Vorheriges Paket: [08b-file-watcher.md](08b-file-watcher.md) — File Watcher + CLI watch (H2)
- Naechstes Paket: —
- Gesamtplan: [08-cli-hooks.md](08-cli-hooks.md)
