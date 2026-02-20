"""FileWatcher — watchdog-based file system watcher with debounce."""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    from collections.abc import Callable


class DebouncedHandler(FileSystemEventHandler):
    """Debounces file system events and calls a callback per unique file.

    Groups rapid consecutive changes to the same file into a single callback
    invocation after debounce_ms milliseconds of quiet time.
    """

    def __init__(
        self,
        callback: Callable[[Path], Any],
        extensions: set[str],
        ignore_dirs: set[str],
        debounce_ms: int = 500,
    ):
        super().__init__()
        self.callback = callback
        self.extensions = extensions  # z.B. {".py", ".ts"}
        self.ignore_dirs = ignore_dirs
        self.debounce_s = debounce_ms / 1000.0
        self._pending: dict[str, float] = {}  # path -> last_event_time
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        p = Path(path)
        for part in p.parts:
            if part in self.ignore_dirs:
                return True
        return p.suffix not in self.extensions

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path
        if self._should_ignore(src_path):
            return

        with self._lock:
            self._pending[src_path] = time.monotonic()

        self._schedule_flush()

    def _schedule_flush(self) -> None:
        """Schedule a flush after debounce period."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce_s, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        """Flush all pending changes."""
        with self._lock:
            paths = list(self._pending.keys())
            self._pending.clear()

        for path_str in paths:
            path = Path(path_str)
            if path.exists():  # skip deleted files in callback
                with contextlib.suppress(Exception):
                    self.callback(path)


class FileWatcher:
    """Watches a directory for code file changes.

    Uses watchdog Observer with DebouncedHandler to detect changes
    and trigger reindexing of modified files.

    Args:
        root: Root directory to watch.
        callback: Function called with Path of each changed file.
        extensions: Set of file extensions to watch (e.g. {".py", ".ts"}).
        ignore_dirs: Set of directory names to ignore.
        debounce_ms: Debounce period in milliseconds.
    """

    def __init__(
        self,
        root: Path,
        callback: Callable[[Path], Any],
        extensions: set[str],
        ignore_dirs: set[str] | None = None,
        debounce_ms: int = 500,
    ):
        self.root = Path(root)
        self.callback = callback
        self.extensions = extensions
        self.ignore_dirs = ignore_dirs or set()
        self.debounce_ms = debounce_ms
        self._observer: Observer | None = None
        self._handler: DebouncedHandler | None = None
        self._running = False

    def start(self) -> None:
        """Start watching the directory."""
        if self._running:
            return

        self._handler = DebouncedHandler(
            callback=self.callback,
            extensions=self.extensions,
            ignore_dirs=self.ignore_dirs,
            debounce_ms=self.debounce_ms,
        )
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.root), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._running = True

    def stop(self) -> None:
        """Stop watching."""
        if not self._running:
            return
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        self._running = False

    @property
    def is_running(self) -> bool:
        """Return whether the watcher is currently running."""
        return self._running

    def __enter__(self) -> FileWatcher:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
