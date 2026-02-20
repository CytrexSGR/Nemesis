"""Indexing Pipeline — Single File Index, Reindex, Full & Delta Project Index."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemesis.graph.adapter import EdgeData, NodeData
from nemesis.indexer.chunker import chunk_node
from nemesis.indexer.delta import (
    DEFAULT_IGNORE_DIRS,
    _collect_code_files,
    _get_extensions,
    delete_file_data,
    detect_changes,
)
from nemesis.indexer.models import ChangeType, IndexResult
from nemesis.parser.models import CodeEdge, CodeNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemesis.graph.adapter import GraphAdapter
    from nemesis.vector.embeddings import EmbeddingProvider
    from nemesis.vector.store import VectorStore


def _prefix_id(project_id: str, original_id: str) -> str:
    """Add project prefix to a node/edge ID.

    When *project_id* is empty the original ID is returned unchanged,
    keeping the pipeline fully backward-compatible.
    """
    if project_id and not original_id.startswith(f"{project_id}::"):
        return f"{project_id}::{original_id}"
    return original_id


def _to_node_data(node: CodeNode | NodeData | Any) -> NodeData | Any:
    """Convert a CodeNode to NodeData for the graph adapter. Pass others through."""
    if isinstance(node, CodeNode):
        props: dict[str, object] = {
            "name": node.name,
            "file": node.file,
            "line_start": node.line_start,
            "line_end": node.line_end,
            "language": node.language,
        }
        # File nodes use "path" instead of "file" in the graph schema.
        if node.kind == "File":
            props["path"] = node.file
        if node.docstring:
            props["docstring"] = node.docstring
        if node.signature:
            props["signature"] = node.signature
        if node.source:
            props["source"] = node.source
        return NodeData(id=node.id, node_type=node.kind, properties=props)
    return node


def _to_edge_data(edge: CodeEdge | EdgeData | Any) -> EdgeData | Any:
    """Convert a CodeEdge to EdgeData for the graph adapter. Pass others through."""
    if isinstance(edge, CodeEdge):
        return EdgeData(
            source_id=edge.source_id,
            target_id=edge.target_id,
            edge_type=edge.kind,
            properties={"file": edge.file},
        )
    return edge


class IndexingPipeline:
    """Pipeline zum Indexieren einzelner Dateien.

    Parst eine Datei, speichert Knoten und Kanten im Graph,
    chunked grosse Knoten, erstellt Embeddings und speichert
    diese im Vector Store.

    Args:
        parser: Parser mit ``parse_file(path)``-Methode.
        graph: Graph-Adapter fuer Knoten/Kanten.
        vector_store: Vector Store fuer Embeddings.
        embedder: Embedding-Provider fuer Texte.
        max_tokens_per_chunk: Max Tokens pro Chunk (default 500).
        on_progress: Optionaler Callback ``(step: str, detail: str) -> None``.
    """

    def __init__(
        self,
        parser: Any,
        graph: GraphAdapter,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        max_tokens_per_chunk: int = 500,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> None:
        self.parser = parser
        self.graph = graph
        self.vector_store = vector_store
        self.embedder = embedder
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.on_progress = on_progress

    def _notify(self, step: str, detail: str) -> None:
        """Ruft den on_progress Callback auf, falls vorhanden."""
        if self.on_progress is not None:
            self.on_progress(step, detail)

    def index_file(
        self,
        path: Path,
        project_id: str = "",
        project_root: Path | None = None,
    ) -> IndexResult:
        """Indexiert eine einzelne Datei.

        Ablauf: Parse -> Knoten/Kanten speichern -> Chunken ->
        Embeddings erstellen -> Embeddings speichern.

        Args:
            path: Pfad zur zu indexierenden Datei.
            project_id: Optionale Projekt-ID fuer Multi-Projekt-Support.
                Wenn angegeben, werden alle Node/Edge IDs mit
                ``project_id::`` geprefixed.
            project_root: Wurzelverzeichnis des Projekts. Wenn angegeben,
                werden Dateipfade relativ zu diesem Verzeichnis gespeichert.

        Returns:
            IndexResult mit Statistiken und ggf. Fehlern.
        """
        start = time.monotonic()
        errors: list[str] = []
        nodes_created = 0
        edges_created = 0
        chunks_created = 0
        embeddings_created = 0

        # Relativen Pfad berechnen (falls project_root gesetzt)
        rel_path = path.relative_to(project_root) if project_root else path

        try:
            # 1. Parse
            self._notify("parse", str(path))
            parse_result = self.parser.parse_file(str(path))

            # 1b. Project-ID Prefix und relative Pfade anwenden
            for node in parse_result.nodes:
                node.id = _prefix_id(project_id, node.id)
                if project_root and hasattr(node, "file"):
                    node.file = str(rel_path)
            for edge in parse_result.edges:
                edge.source_id = _prefix_id(project_id, edge.source_id)
                edge.target_id = _prefix_id(project_id, edge.target_id)
                if project_root and hasattr(edge, "file"):
                    edge.file = str(rel_path)

            # 2. Knoten im Graph speichern
            self._notify("store_nodes", f"{len(parse_result.nodes)} nodes")
            for node in parse_result.nodes:
                try:
                    self.graph.add_node(_to_node_data(node))
                    nodes_created += 1
                except Exception as e:
                    errors.append(f"add_node failed for {getattr(node, 'id', '?')}: {e}")

            # 3. Kanten im Graph speichern
            self._notify("store_edges", f"{len(parse_result.edges)} edges")
            for edge in parse_result.edges:
                try:
                    self.graph.add_edge(_to_edge_data(edge))
                    edges_created += 1
                except Exception as e:
                    errors.append(f"add_edge failed for edge: {e}")

            # 4. Chunken
            self._notify("chunk", "chunking nodes")
            all_chunks = []
            for node in parse_result.nodes:
                source = getattr(node, "source", "")
                if not source:
                    continue
                chunks = chunk_node(
                    node,
                    source,
                    max_tokens=self.max_tokens_per_chunk,
                    file_path=path,
                )
                all_chunks.extend(chunks)
            chunks_created = len(all_chunks)

            # 5. Embeddings erstellen und speichern
            if all_chunks:
                self._notify("embed", f"{len(all_chunks)} chunks")
                texts = [c.content for c in all_chunks]
                try:
                    result = self.embedder.embed(texts)
                    embeddings = result.embeddings

                    # Im Vector Store speichern
                    ids = [c.id for c in all_chunks]
                    metadata = [
                        {
                            "file": str(rel_path) if project_root else str(path),
                            "parent_node_id": c.parent_node_id,
                            "parent_type": c.parent_type,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                        }
                        for c in all_chunks
                    ]
                    self.vector_store.add(
                        ids, texts, embeddings, metadata,
                        project_id=project_id,
                    )
                    embeddings_created = len(all_chunks)
                except Exception as e:
                    errors.append(f"embedding/store failed: {e}")

        except Exception as e:
            errors.append(f"index_file failed for {path}: {e}")

        elapsed_ms = (time.monotonic() - start) * 1000
        return IndexResult(
            files_indexed=1 if not errors else 0,
            nodes_created=nodes_created,
            edges_created=edges_created,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            duration_ms=elapsed_ms,
            errors=errors,
        )

    def reindex_file(
        self,
        path: Path,
        project_id: str = "",
        project_root: Path | None = None,
    ) -> IndexResult:
        """Re-indexiert eine Datei (Delta Update).

        Loescht zuerst alle alten Daten der Datei aus Graph und
        Vector Store und indexiert sie dann neu.

        Args:
            path: Pfad zur zu re-indexierenden Datei.
            project_id: Optionale Projekt-ID (weitergereicht an index_file).
            project_root: Optionales Wurzelverzeichnis (weitergereicht an index_file).

        Returns:
            IndexResult mit Statistiken und ggf. Fehlern.
        """
        start = time.monotonic()
        # Use relative path for cleanup if project_root is set,
        # because nodes are stored with relative paths in the graph.
        lookup_path = str(path.relative_to(project_root)) if project_root else str(path)

        # Cleanup-Phase: alte Daten loeschen
        try:
            self._notify("cleanup", f"deleting old data for {lookup_path}")
            chunk_ids = self.graph.get_chunk_ids_for_file(lookup_path)
            self.graph.delete_nodes_for_file(lookup_path)
            if chunk_ids:
                self.vector_store.delete_embeddings(chunk_ids)
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            return IndexResult(
                files_indexed=0,
                nodes_created=0,
                edges_created=0,
                chunks_created=0,
                embeddings_created=0,
                duration_ms=elapsed_ms,
                errors=[f"cleanup failed for {lookup_path}: {e}"],
            )

        # Neu-Indexierung
        return self.index_file(path, project_id=project_id, project_root=project_root)

    # ------------------------------------------------------------------
    # Task 9: Full Project Index
    # ------------------------------------------------------------------

    def index_project(
        self,
        path: Path,
        languages: list[str],
        ignore_dirs: set[str] | None = None,
        project_id: str = "",
        project_root: Path | None = None,
    ) -> IndexResult:
        """Indexiert ein gesamtes Projekt.

        Sammelt alle Code-Dateien fuer die angegebenen Sprachen
        und indexiert jede einzeln. Fehler einzelner Dateien werden
        gesammelt, ohne den Gesamtprozess abzubrechen.

        Args:
            path: Wurzelverzeichnis des Projekts.
            languages: Liste von Sprachen (z.B. ``["python"]``).
            ignore_dirs: Zusaetzliche Verzeichnisnamen zum Ignorieren.
            project_id: Optionale Projekt-ID fuer Multi-Projekt-Support.
            project_root: Wurzelverzeichnis fuer relative Pfade.
                Wenn nicht angegeben, wird *path* verwendet.

        Returns:
            Aggregiertes IndexResult ueber alle Dateien.
        """
        start = time.monotonic()
        effective_ignore = DEFAULT_IGNORE_DIRS | (ignore_dirs or set())
        extensions = _get_extensions(languages)
        files = _collect_code_files(Path(path), extensions, effective_ignore)

        effective_root = project_root or path

        # Alte Vektor-Daten loeschen falls bereits indexiert (Duplikat-Vermeidung).
        # Graph nutzt MERGE (Upsert) und hat keine Duplikate.
        if project_id:
            try:
                self.vector_store.delete_by_project(project_id)
                self._notify("cleanup", f"cleared old embeddings for {project_id}")
            except Exception:
                pass

        total_files_indexed = 0
        total_nodes = 0
        total_edges = 0
        total_chunks = 0
        total_embeddings = 0
        all_errors: list[str] = []

        for i, file_path in enumerate(files):
            self._notify(
                "index_project",
                f"[{i + 1}/{len(files)}] {file_path}",
            )
            try:
                result = self.index_file(
                    file_path,
                    project_id=project_id,
                    project_root=effective_root,
                )
                total_files_indexed += result.files_indexed
                total_nodes += result.nodes_created
                total_edges += result.edges_created
                total_chunks += result.chunks_created
                total_embeddings += result.embeddings_created
                all_errors.extend(result.errors)
            except Exception as e:
                all_errors.append(f"index_project failed for {file_path}: {e}")

        # Auto-Kompaktierung der Vektor-DB
        try:
            self._notify("optimize", "compacting vector store")
            self.vector_store.optimize()
        except Exception as e:
            all_errors.append(f"vector store optimize failed: {e}")

        elapsed_ms = (time.monotonic() - start) * 1000
        return IndexResult(
            files_indexed=total_files_indexed,
            nodes_created=total_nodes,
            edges_created=total_edges,
            chunks_created=total_chunks,
            embeddings_created=total_embeddings,
            duration_ms=elapsed_ms,
            errors=all_errors,
        )

    # ------------------------------------------------------------------
    # Task 10: Delta Project Update
    # ------------------------------------------------------------------

    def update_project(
        self,
        path: Path,
        languages: list[str],
        ignore_dirs: set[str] | None = None,
        project_id: str = "",
        project_root: Path | None = None,
    ) -> IndexResult:
        """Fuehrt ein Delta-Update fuer ein Projekt durch.

        Erkennt Aenderungen gegenueber dem Graph-Stand und
        verarbeitet nur geaenderte, neue und geloeschte Dateien.

        Args:
            path: Wurzelverzeichnis des Projekts.
            languages: Liste von Sprachen.
            ignore_dirs: Zusaetzliche Verzeichnisnamen zum Ignorieren.
            project_id: Optionale Projekt-ID fuer Multi-Projekt-Support.
            project_root: Wurzelverzeichnis fuer relative Pfade.
                Wenn nicht angegeben, wird *path* verwendet.

        Returns:
            Aggregiertes IndexResult ueber alle Aenderungen.
        """
        start = time.monotonic()
        changes = detect_changes(
            Path(path),
            self.graph,
            languages,
            ignore_dirs,
        )

        effective_root = project_root or path

        total_files_indexed = 0
        total_nodes = 0
        total_edges = 0
        total_chunks = 0
        total_embeddings = 0
        all_errors: list[str] = []

        for i, change in enumerate(changes):
            self._notify(
                "update_project",
                f"[{i + 1}/{len(changes)}] {change.change_type.value}: {change.path}",
            )
            try:
                if change.change_type == ChangeType.DELETED:
                    # Use relative path for deletion if project_root is set
                    del_path = str(change.path.relative_to(effective_root)) if project_root else str(change.path)
                    delete_file_data(
                        del_path,
                        self.graph,
                        self.vector_store,
                    )
                    total_files_indexed += 1
                elif change.change_type == ChangeType.MODIFIED:
                    result = self.reindex_file(
                        change.path,
                        project_id=project_id,
                        project_root=effective_root,
                    )
                    total_files_indexed += result.files_indexed
                    total_nodes += result.nodes_created
                    total_edges += result.edges_created
                    total_chunks += result.chunks_created
                    total_embeddings += result.embeddings_created
                    all_errors.extend(result.errors)
                elif change.change_type == ChangeType.ADDED:
                    result = self.index_file(
                        change.path,
                        project_id=project_id,
                        project_root=effective_root,
                    )
                    total_files_indexed += result.files_indexed
                    total_nodes += result.nodes_created
                    total_edges += result.edges_created
                    total_chunks += result.chunks_created
                    total_embeddings += result.embeddings_created
                    all_errors.extend(result.errors)
            except Exception as e:
                all_errors.append(f"update_project failed for {change.path}: {e}")

        # Auto-Kompaktierung der Vektor-DB
        if changes:
            try:
                self._notify("optimize", "compacting vector store")
                self.vector_store.optimize()
            except Exception as e:
                all_errors.append(f"vector store optimize failed: {e}")

        elapsed_ms = (time.monotonic() - start) * 1000
        return IndexResult(
            files_indexed=total_files_indexed,
            nodes_created=total_nodes,
            edges_created=total_edges,
            chunks_created=total_chunks,
            embeddings_created=total_embeddings,
            duration_ms=elapsed_ms,
            errors=all_errors,
        )
