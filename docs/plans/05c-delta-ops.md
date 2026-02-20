# 05c — Delta Detection + Deletion

> **Arbeitspaket E3** — Teil 3 von 5 des Indexing Pipeline Plans

**Goal:** Delta-Erkennung durch File-Hash-Vergleich (neue, geaenderte, geloeschte Dateien) sowie das Loeschen veralteter Daten aus Graph und Vector Store.

**Tech Stack:** Python, hashlib (SHA-256), nemesis.indexer.models (FileChange, ChangeType)

**Abhängigkeiten:** E1 (05a-indexer-models)

**Tasks in diesem Paket:** 5, 6

---

## Task 5: Delta Detection — File-Hashes vergleichen

**Files:**
- `nemesis/indexer/delta.py`
- `tests/test_indexer/test_delta.py`

### Step 1 — Write Test

```python
# tests/test_indexer/test_delta.py
"""Tests fuer Delta/Diff-Erkennung."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import hashlib


def _make_mock_graph():
    """Erzeugt einen Mock-GraphAdapter."""
    graph = MagicMock()
    return graph


def test_compute_file_hash():
    """File-Hash wird korrekt berechnet."""
    from nemesis.indexer.delta import compute_file_hash

    import tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def hello():\n    pass\n")
        f.flush()
        path = Path(f.name)

    try:
        h = compute_file_hash(path)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
    finally:
        os.unlink(path)


def test_compute_file_hash_deterministic():
    """Gleicher Inhalt ergibt gleichen Hash."""
    from nemesis.indexer.delta import compute_file_hash

    import tempfile, os

    content = "x = 42\n"
    paths = []
    for _ in range(2):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            paths.append(Path(f.name))

    try:
        assert compute_file_hash(paths[0]) == compute_file_hash(paths[1])
    finally:
        for p in paths:
            os.unlink(p)


def test_compute_file_hash_different_content():
    """Unterschiedlicher Inhalt ergibt unterschiedlichen Hash."""
    from nemesis.indexer.delta import compute_file_hash

    import tempfile, os

    paths = []
    for content in ["x = 1\n", "x = 2\n"]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            paths.append(Path(f.name))

    try:
        assert compute_file_hash(paths[0]) != compute_file_hash(paths[1])
    finally:
        for p in paths:
            os.unlink(p)


def test_detect_changes_new_file(tmp_path):
    """Neue Datei wird als ADDED erkannt."""
    from nemesis.indexer.delta import detect_changes
    from nemesis.indexer.models import ChangeType

    # Erstelle eine Python-Datei
    py_file = tmp_path / "new.py"
    py_file.write_text("print('hello')\n")

    # Graph hat keine File-Nodes -> alles ist neu
    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {}

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.ADDED
    assert changes[0].path == py_file
    assert changes[0].old_hash is None
    assert changes[0].new_hash is not None


def test_detect_changes_modified_file(tmp_path):
    """Geaenderte Datei wird als MODIFIED erkannt."""
    from nemesis.indexer.delta import detect_changes
    from nemesis.indexer.models import ChangeType

    py_file = tmp_path / "existing.py"
    py_file.write_text("x = 2\n")

    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {
        str(py_file): "old_hash_value",
    }

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.MODIFIED
    assert changes[0].old_hash == "old_hash_value"


def test_detect_changes_deleted_file(tmp_path):
    """Geloeschte Datei wird als DELETED erkannt."""
    from nemesis.indexer.delta import detect_changes
    from nemesis.indexer.models import ChangeType

    # Graph kennt eine Datei die nicht mehr existiert
    graph = _make_mock_graph()
    deleted_path = str(tmp_path / "deleted.py")
    graph.get_file_hashes.return_value = {
        deleted_path: "some_hash",
    }

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.DELETED
    assert changes[0].path == Path(deleted_path)
    assert changes[0].new_hash is None


def test_detect_changes_unchanged_file(tmp_path):
    """Unveraenderte Datei erzeugt keinen Change."""
    from nemesis.indexer.delta import detect_changes, compute_file_hash

    py_file = tmp_path / "stable.py"
    py_file.write_text("x = 1\n")

    current_hash = compute_file_hash(py_file)

    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {
        str(py_file): current_hash,
    }

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 0


def test_detect_changes_ignores_non_code_files(tmp_path):
    """Nicht-Code-Dateien werden ignoriert."""
    from nemesis.indexer.delta import detect_changes

    (tmp_path / "readme.md").write_text("# Hello\n")
    (tmp_path / "data.json").write_text("{}\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {}

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 0


def test_detect_changes_multiple_languages(tmp_path):
    """Erkennung funktioniert mit mehreren Sprachen."""
    from nemesis.indexer.delta import detect_changes

    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "index.ts").write_text("const x = 1;\n")

    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {}

    changes = detect_changes(tmp_path, graph, languages=["python", "typescript"])
    assert len(changes) == 2
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_delta.py -v
# Erwartung: FAILED — Module nemesis.indexer.delta existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/delta.py
"""Delta/Diff-Erkennung fuer inkrementelle Index-Updates.

Vergleicht File-Hashes auf Disk mit den im Graph gespeicherten Hashes.
Erkennt neue, geaenderte und geloeschte Dateien.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from nemesis.indexer.models import ChangeType, FileChange


# Dateiendungen pro Sprache
LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py", ".pyi"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx", ".mjs"],
    "rust": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx"],
    "ruby": [".rb"],
    "php": [".php"],
}

# Standard-Verzeichnisse die ignoriert werden
DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "target",
    "build",
    "dist",
    ".nemesis",
}


def compute_file_hash(path: Path) -> str:
    """Berechne SHA-256 Hash einer Datei.

    Args:
        path: Pfad zur Datei.

    Returns:
        Hex-String des SHA-256 Hashes.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _get_extensions(languages: list[str]) -> set[str]:
    """Sammle alle relevanten Dateiendungen fuer die angegebenen Sprachen."""
    extensions: set[str] = set()
    for lang in languages:
        lang_lower = lang.lower()
        if lang_lower in LANGUAGE_EXTENSIONS:
            extensions.update(LANGUAGE_EXTENSIONS[lang_lower])
    return extensions


def _collect_code_files(
    project_path: Path,
    extensions: set[str],
    ignore_dirs: set[str] | None = None,
) -> list[Path]:
    """Finde alle Code-Dateien im Projekt rekursiv.

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        extensions: Erlaubte Dateiendungen.
        ignore_dirs: Verzeichnisnamen die uebersprungen werden.

    Returns:
        Liste der gefundenen Code-Dateien.
    """
    if ignore_dirs is None:
        ignore_dirs = DEFAULT_IGNORE_DIRS

    files: list[Path] = []
    for item in sorted(project_path.rglob("*")):
        if item.is_file() and item.suffix in extensions:
            # Pruefen ob ein Eltern-Verzeichnis ignoriert werden soll
            parts = item.relative_to(project_path).parts
            if not any(part in ignore_dirs for part in parts):
                files.append(item)
    return files


def detect_changes(
    project_path: Path,
    graph: Any,
    languages: list[str],
    ignore_dirs: set[str] | None = None,
) -> list[FileChange]:
    """Erkenne Datei-Aenderungen seit dem letzten Index.

    Vergleicht aktuelle Dateien auf Disk mit den im Graph
    gespeicherten File-Hashes. Erkennt:
    - ADDED: Datei existiert auf Disk, aber nicht im Graph
    - MODIFIED: Datei existiert in beiden, aber Hash unterschiedlich
    - DELETED: Datei existiert im Graph, aber nicht mehr auf Disk

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        graph: GraphAdapter mit get_file_hashes() Methode.
        languages: Liste der zu indexierenden Sprachen.
        ignore_dirs: Verzeichnisnamen die ignoriert werden.

    Returns:
        Liste von FileChange-Objekten.
    """
    extensions = _get_extensions(languages)
    current_files = _collect_code_files(project_path, extensions, ignore_dirs)

    # Aktuelle Hashes berechnen
    current_hashes: dict[str, str] = {}
    for file_path in current_files:
        current_hashes[str(file_path)] = compute_file_hash(file_path)

    # Gespeicherte Hashes aus dem Graph
    stored_hashes: dict[str, str] = graph.get_file_hashes()

    changes: list[FileChange] = []

    # Neue und geaenderte Dateien
    for path_str, new_hash in sorted(current_hashes.items()):
        if path_str not in stored_hashes:
            changes.append(
                FileChange(
                    path=Path(path_str),
                    change_type=ChangeType.ADDED,
                    old_hash=None,
                    new_hash=new_hash,
                )
            )
        elif stored_hashes[path_str] != new_hash:
            changes.append(
                FileChange(
                    path=Path(path_str),
                    change_type=ChangeType.MODIFIED,
                    old_hash=stored_hashes[path_str],
                    new_hash=new_hash,
                )
            )

    # Geloeschte Dateien
    for path_str, old_hash in sorted(stored_hashes.items()):
        if path_str not in current_hashes:
            changes.append(
                FileChange(
                    path=Path(path_str),
                    change_type=ChangeType.DELETED,
                    old_hash=old_hash,
                    new_hash=None,
                )
            )

    return changes
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_delta.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/delta.py tests/test_indexer/test_delta.py
git commit -m "feat(indexer): add delta detection with file hash comparison"
```

---

## Task 6: Delta — Alte Daten loeschen

**Files:**
- `nemesis/indexer/delta.py` (erweitern)
- `tests/test_indexer/test_delta.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_delta.py — APPEND folgende Tests

def test_delete_file_data_calls_graph_and_vector():
    """delete_file_data loescht Nodes, Edges und Embeddings."""
    from nemesis.indexer.delta import delete_file_data

    graph = _make_mock_graph()
    graph.get_nodes_for_file.return_value = [
        {"id": "func-001", "type": "Function"},
        {"id": "func-002", "type": "Function"},
    ]
    graph.get_chunk_ids_for_file.return_value = ["chunk-001", "chunk-002"]

    vector_store = MagicMock()

    file_path = Path("/project/src/old.py")
    delete_file_data(file_path, graph, vector_store)

    # Graph: Nodes und Edges fuer die Datei loeschen
    graph.delete_nodes_for_file.assert_called_once_with(str(file_path))
    # Vector Store: Embeddings fuer die Chunks loeschen
    vector_store.delete_embeddings.assert_called_once_with(["chunk-001", "chunk-002"])


def test_delete_file_data_no_chunks():
    """delete_file_data funktioniert wenn keine Chunks existieren."""
    from nemesis.indexer.delta import delete_file_data

    graph = _make_mock_graph()
    graph.get_nodes_for_file.return_value = [{"id": "func-001", "type": "Function"}]
    graph.get_chunk_ids_for_file.return_value = []

    vector_store = MagicMock()

    delete_file_data(Path("/project/src/empty.py"), graph, vector_store)

    graph.delete_nodes_for_file.assert_called_once()
    vector_store.delete_embeddings.assert_called_once_with([])


def test_delete_file_data_no_existing_data():
    """delete_file_data funktioniert wenn Datei nicht im Graph ist."""
    from nemesis.indexer.delta import delete_file_data

    graph = _make_mock_graph()
    graph.get_nodes_for_file.return_value = []
    graph.get_chunk_ids_for_file.return_value = []

    vector_store = MagicMock()

    delete_file_data(Path("/project/src/unknown.py"), graph, vector_store)

    graph.delete_nodes_for_file.assert_called_once()
    vector_store.delete_embeddings.assert_called_once_with([])
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_delta.py -v -k "delete_file"
# Erwartung: FAILED — delete_file_data existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/delta.py — APPEND am Ende der Datei

def delete_file_data(
    file_path: Path,
    graph: Any,
    vector_store: Any,
) -> None:
    """Loesche alle Daten einer Datei aus Graph und Vector Store.

    Wird vor dem Re-Index einer geaenderten Datei aufgerufen,
    damit keine veralteten Nodes/Embeddings zurueckbleiben.

    Args:
        file_path: Pfad der Datei deren Daten geloescht werden.
        graph: GraphAdapter mit delete_nodes_for_file() und
               get_chunk_ids_for_file() Methoden.
        vector_store: VectorStore mit delete_embeddings() Methode.
    """
    # Chunk-IDs sammeln bevor die Nodes geloescht werden
    chunk_ids: list[str] = graph.get_chunk_ids_for_file(str(file_path))

    # Graph: Alle Nodes und Edges fuer diese Datei loeschen
    graph.delete_nodes_for_file(str(file_path))

    # Vector Store: Embeddings fuer die Chunks loeschen
    vector_store.delete_embeddings(chunk_ids)
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_delta.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/delta.py tests/test_indexer/test_delta.py
git commit -m "feat(indexer): add delete_file_data for cleaning stale graph/vector data"
```

---

## Zusammenfassung E3

| Task | Datei(en) | Beschreibung |
|------|-----------|-------------|
| 5 | `delta.py` | Delta Detection: File-Hashes vergleichen, Changes erkennen |
| 6 | `delta.py` | delete_file_data: Alte Graph/Vector-Daten loeschen |

---

**Navigation:**
- Zurueck: [05b — Chunking (E2)](05b-chunking.md)
- Weiter: [05d — Single File Pipeline (E4)](05d-single-file-pipeline.md)
