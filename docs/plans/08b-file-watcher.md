# File Watcher + CLI watch

> **Arbeitspaket H2** — Teil 2 von 3 des CLI & Hooks Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the file watcher system with proper debouncing for incremental re-indexing, implement the NemesisWatcher lifecycle, and integrate it into the CLI `nemesis watch` command.

**Architecture:** The file watcher (`nemesis/core/watcher.py`) uses the watchdog library with proper debouncing to trigger incremental re-indexing. The `DebouncedHandler` collects rapid events per file and fires a callback only after the debounce period. The `NemesisWatcher` manages the Observer lifecycle. The CLI `watch` command ties everything together with signal handling.

**Tech Stack:** Python 3.11+, Click, watchdog, pytest

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [08a-cli-commands.md](08a-cli-commands.md) (Task 1-3 fuer `_check_initialized` und CLI-Grundstruktur)

**Tasks in this package:** 4, 5, 6 (von 10)

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

## Zusammenfassung (Arbeitspaket H2)

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 4 | DebouncedHandler mit per-file Debouncing | `watcher.py` | 8 |
| 5 | NemesisWatcher Integration (start/stop/lifecycle) | `watcher.py` | 7 |
| 6 | `nemesis watch` CLI Integration | `cli.py` | 3 |
| **Gesamt H2** | | **2 Impl-Dateien, 2 Test-Dateien** | **~18 Tests** |

---

**Navigation:**
- Vorheriges Paket: [08a-cli-commands.md](08a-cli-commands.md) — CLI: init + index/status + query/rule (H1)
- Naechstes Paket: [08c-hooks-system.md](08c-hooks-system.md) — Hooks + Config Generator (H3)
- Gesamtplan: [08-cli-hooks.md](08-cli-hooks.md)
