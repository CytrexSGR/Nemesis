"""Delta Detection — erkennt geaenderte, neue und geloeschte Dateien."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from nemesis.indexer.models import ChangeType, FileChange

if TYPE_CHECKING:
    from nemesis.graph.adapter import GraphAdapter
    from nemesis.vector.store import VectorStore

# ---------------------------------------------------------------------------
# Language extensions — mapping of language name to file suffixes
# ---------------------------------------------------------------------------

LANGUAGE_EXTENSIONS: dict[str, set[str]] = {
    "python": {".py", ".pyi"},
    "typescript": {".ts", ".tsx"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "rust": {".rs"},
    "go": {".go"},
    "java": {".java"},
    "c": {".c", ".h"},
    "cpp": {".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx"},
    "csharp": {".cs"},
    "ruby": {".rb"},
    "php": {".php"},
    "swift": {".swift"},
    "kotlin": {".kt", ".kts"},
    "scala": {".scala"},
    "zig": {".zig"},
}

# ---------------------------------------------------------------------------
# Directories to always skip during file collection
# ---------------------------------------------------------------------------

DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "target",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "egg-info",
    ".eggs",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    """Berechnet den SHA-256-Hash einer Datei.

    Liest die Datei in 8192-Byte-Bloecken, um auch grosse
    Dateien speicherschonend verarbeiten zu koennen.

    Args:
        path: Pfad zur Datei.

    Returns:
        64-Zeichen Hex-String (SHA-256).
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_extensions(languages: list[str]) -> set[str]:
    """Sammelt alle Dateiendungen fuer die angegebenen Sprachen.

    Args:
        languages: Liste von Sprachnamen (z.B. ``["python", "typescript"]``).

    Returns:
        Menge aller zugehoerigen Dateiendungen.
    """
    extensions: set[str] = set()
    for lang in languages:
        key = lang.lower()
        if key in LANGUAGE_EXTENSIONS:
            extensions |= LANGUAGE_EXTENSIONS[key]
    return extensions


def _collect_code_files(
    project_path: Path,
    extensions: set[str],
    ignore_dirs: set[str],
) -> list[Path]:
    """Sammelt alle Code-Dateien im Projektverzeichnis.

    Durchlaeuft das Verzeichnis rekursiv und gibt nur Dateien
    zurueck, deren Suffix in *extensions* enthalten ist.
    Verzeichnisse aus *ignore_dirs* werden uebersprungen.

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        extensions: Erlaubte Dateiendungen.
        ignore_dirs: Verzeichnisnamen, die ignoriert werden sollen.

    Returns:
        Sortierte Liste von Dateipfaden.
    """
    files: list[Path] = []
    for item in sorted(project_path.rglob("*")):
        # Verzeichnisse aus ignore_dirs ueberspringen
        if any(part in ignore_dirs for part in item.parts):
            continue
        if item.is_file() and item.suffix in extensions:
            files.append(item)
    return files


def detect_changes(
    project_path: Path,
    graph: GraphAdapter,
    languages: list[str],
    ignore_dirs: set[str] | None = None,
) -> list[FileChange]:
    """Erkennt Datei-Aenderungen gegenueber dem Graph-Stand.

    Vergleicht die aktuellen Dateien auf der Festplatte mit den
    im Graph gespeicherten Hashes und klassifiziert jede Datei
    als ADDED, MODIFIED oder DELETED.

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        graph: Graph-Adapter mit ``get_file_hashes()``-Methode.
        languages: Liste von Sprachen, deren Dateien betrachtet werden.
        ignore_dirs: Zusaetzliche Verzeichnisnamen zum Ignorieren.
                     Wird mit DEFAULT_IGNORE_DIRS zusammengefuehrt.

    Returns:
        Liste von FileChange-Objekten fuer alle erkannten Aenderungen.
    """
    effective_ignore = DEFAULT_IGNORE_DIRS | (ignore_dirs or set())
    extensions = _get_extensions(languages)
    disk_files = _collect_code_files(project_path, extensions, effective_ignore)

    stored_hashes = graph.get_file_hashes()

    changes: list[FileChange] = []

    # Dateien auf der Festplatte pruefen
    seen_paths: set[str] = set()
    for file_path in disk_files:
        path_str = str(file_path)
        seen_paths.add(path_str)
        current_hash = compute_file_hash(file_path)
        old_hash = stored_hashes.get(path_str)

        if old_hash is None:
            # Neue Datei
            changes.append(
                FileChange(
                    path=file_path,
                    change_type=ChangeType.ADDED,
                    old_hash=None,
                    new_hash=current_hash,
                )
            )
        elif old_hash != current_hash:
            # Geaenderte Datei
            changes.append(
                FileChange(
                    path=file_path,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_hash,
                    new_hash=current_hash,
                )
            )
        # else: unveraendert — kein Eintrag

    # Geloeschte Dateien: im Graph vorhanden, aber nicht mehr auf der Festplatte
    for stored_path in stored_hashes:
        if stored_path not in seen_paths:
            changes.append(
                FileChange(
                    path=Path(stored_path),
                    change_type=ChangeType.DELETED,
                    old_hash=stored_hashes[stored_path],
                    new_hash=None,
                )
            )

    return changes


def delete_file_data(
    file_path: str,
    graph: GraphAdapter,
    vector_store: VectorStore,
) -> None:
    """Loescht alle Daten einer Datei aus Graph und Vector Store.

    Holt zuerst die Chunk-IDs aus dem Graph, loescht dann die
    Knoten/Kanten im Graph und anschliessend die zugehoerigen
    Embeddings im Vector Store.

    Args:
        file_path: Pfad der Datei, deren Daten entfernt werden sollen.
        graph: Graph-Adapter.
        vector_store: Vector Store fuer Embeddings.
    """
    # Chunk-IDs holen, bevor die Knoten geloescht werden
    chunk_ids = graph.get_chunk_ids_for_file(file_path)

    # Knoten und Kanten im Graph loeschen
    graph.delete_nodes_for_file(file_path)

    # Embeddings im Vector Store loeschen (nur wenn Chunks vorhanden)
    if chunk_ids:
        vector_store.delete_embeddings(chunk_ids)
