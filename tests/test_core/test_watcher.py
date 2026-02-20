"""Tests for nemesis.core.watcher — DebouncedHandler and FileWatcher."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from watchdog.events import DirModifiedEvent, FileModifiedEvent

from nemesis.core.watcher import DebouncedHandler, FileWatcher

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# TestDebouncedHandler
# ---------------------------------------------------------------------------


class TestDebouncedHandler:
    """Tests for the DebouncedHandler class."""

    def _make_handler(
        self,
        callback: MagicMock | None = None,
        extensions: set[str] | None = None,
        ignore_dirs: set[str] | None = None,
        debounce_ms: int = 100,
    ) -> tuple[DebouncedHandler, MagicMock]:
        cb = callback or MagicMock()
        handler = DebouncedHandler(
            callback=cb,
            extensions=extensions or {".py"},
            ignore_dirs=ignore_dirs or {"__pycache__", ".git"},
            debounce_ms=debounce_ms,
        )
        return handler, cb

    def test_ignores_directory_events(self) -> None:
        """Directory events werden ignoriert."""
        handler, cb = self._make_handler()
        event = DirModifiedEvent(src_path="/some/dir")
        handler.on_any_event(event)
        # Kein pending event registriert
        assert len(handler._pending) == 0
        time.sleep(0.2)
        cb.assert_not_called()

    def test_ignores_wrong_extensions(self) -> None:
        """Dateien mit falschen Extensions werden ignoriert."""
        handler, cb = self._make_handler(extensions={".py"})
        event = FileModifiedEvent(src_path="/some/file.txt")
        handler.on_any_event(event)
        assert len(handler._pending) == 0
        time.sleep(0.2)
        cb.assert_not_called()

    def test_ignores_paths_in_ignore_dirs(self) -> None:
        """Dateien in ignore_dirs werden ignoriert."""
        handler, cb = self._make_handler(ignore_dirs={"__pycache__"})
        event = FileModifiedEvent(src_path="/project/__pycache__/module.py")
        handler.on_any_event(event)
        assert len(handler._pending) == 0
        time.sleep(0.2)
        cb.assert_not_called()

    def test_accepts_matching_extension(self, tmp_path: Path) -> None:
        """Eine .py Datei loest den Callback aus."""
        test_file = tmp_path / "module.py"
        test_file.write_text("# code")

        handler, cb = self._make_handler(debounce_ms=50)
        event = FileModifiedEvent(src_path=str(test_file))
        handler.on_any_event(event)

        # Warten bis debounce + flush
        time.sleep(0.3)
        cb.assert_called_once_with(test_file)

    def test_debounce_groups_rapid_events(self, tmp_path: Path) -> None:
        """3 schnelle Events auf gleiche Datei fuehren zu nur 1 Callback."""
        test_file = tmp_path / "module.py"
        test_file.write_text("# v1")

        handler, cb = self._make_handler(debounce_ms=100)

        # 3 schnelle Events hintereinander (innerhalb der debounce-Zeit)
        for i in range(3):
            test_file.write_text(f"# v{i + 1}")
            event = FileModifiedEvent(src_path=str(test_file))
            handler.on_any_event(event)
            time.sleep(0.02)  # 20ms zwischen Events — deutlich unter debounce

        # Warten bis debounce abgelaufen und flush passiert
        time.sleep(0.4)
        cb.assert_called_once_with(test_file)


# ---------------------------------------------------------------------------
# TestFileWatcher
# ---------------------------------------------------------------------------


class TestFileWatcher:
    """Tests for the FileWatcher class."""

    def test_start_stop(self, tmp_path: Path) -> None:
        """start/stop lifecycle funktioniert ohne Fehler."""
        watcher = FileWatcher(
            root=tmp_path,
            callback=lambda p: None,
            extensions={".py"},
            debounce_ms=50,
        )
        watcher.start()
        assert watcher._observer is not None
        assert watcher._observer.is_alive()
        watcher.stop()
        assert not watcher._observer.is_alive()

    def test_is_running_property(self, tmp_path: Path) -> None:
        """is_running ist False vor start, True nach start, False nach stop."""
        watcher = FileWatcher(
            root=tmp_path,
            callback=lambda p: None,
            extensions={".py"},
            debounce_ms=50,
        )
        assert watcher.is_running is False
        watcher.start()
        assert watcher.is_running is True
        watcher.stop()
        assert watcher.is_running is False

    def test_context_manager(self, tmp_path: Path) -> None:
        """with-Statement startet und stoppt den Watcher korrekt."""
        watcher = FileWatcher(
            root=tmp_path,
            callback=lambda p: None,
            extensions={".py"},
            debounce_ms=50,
        )
        with watcher:
            assert watcher.is_running is True
        assert watcher.is_running is False

    def test_double_start_noop(self, tmp_path: Path) -> None:
        """Zweiter start() Aufruf ist idempotent — kein Fehler, gleicher Observer."""
        watcher = FileWatcher(
            root=tmp_path,
            callback=lambda p: None,
            extensions={".py"},
            debounce_ms=50,
        )
        watcher.start()
        first_observer = watcher._observer
        watcher.start()  # zweiter Aufruf
        assert watcher._observer is first_observer  # gleicher Observer
        assert watcher.is_running is True
        watcher.stop()

    def test_detects_file_change(self, tmp_path: Path) -> None:
        """Echte Datei-Aenderung wird erkannt und Callback aufgerufen."""
        callback_event = threading.Event()
        changed_paths: list[Path] = []

        def on_change(path: Path) -> None:
            changed_paths.append(path)
            callback_event.set()

        test_file = tmp_path / "main.py"
        test_file.write_text("# initial")

        watcher = FileWatcher(
            root=tmp_path,
            callback=on_change,
            extensions={".py"},
            debounce_ms=50,
        )
        watcher.start()

        # Kurz warten, damit der Observer aktiv ist
        time.sleep(0.1)

        # Datei aendern
        test_file.write_text("# changed content")

        # Auf Callback warten (max 3 Sekunden)
        callback_event.wait(timeout=3.0)

        watcher.stop()

        assert len(changed_paths) >= 1
        assert test_file in changed_paths
