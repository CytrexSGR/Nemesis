"""Indexing Pipeline — Single File Index + Reindex."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from nemesis.indexer.chunker import chunk_node
from nemesis.indexer.models import IndexResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from nemesis.graph.adapter import GraphAdapter
    from nemesis.vector.embeddings import EmbeddingProvider
    from nemesis.vector.store import VectorStore


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

    def index_file(self, path: Path) -> IndexResult:
        """Indexiert eine einzelne Datei.

        Ablauf: Parse -> Knoten/Kanten speichern -> Chunken ->
        Embeddings erstellen -> Embeddings speichern.

        Args:
            path: Pfad zur zu indexierenden Datei.

        Returns:
            IndexResult mit Statistiken und ggf. Fehlern.
        """
        start = time.monotonic()
        errors: list[str] = []
        nodes_created = 0
        edges_created = 0
        chunks_created = 0
        embeddings_created = 0

        try:
            # 1. Parse
            self._notify("parse", str(path))
            parse_result = self.parser.parse_file(str(path))

            # 2. Knoten im Graph speichern
            self._notify("store_nodes", f"{len(parse_result.nodes)} nodes")
            for node in parse_result.nodes:
                try:
                    self.graph.add_node(node)
                    nodes_created += 1
                except Exception as e:
                    errors.append(f"add_node failed for {getattr(node, 'id', '?')}: {e}")

            # 3. Kanten im Graph speichern
            self._notify("store_edges", f"{len(parse_result.edges)} edges")
            for edge in parse_result.edges:
                try:
                    self.graph.add_edge(edge)
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
                            "file": str(path),
                            "parent_node_id": c.parent_node_id,
                            "parent_type": c.parent_type,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                        }
                        for c in all_chunks
                    ]
                    self.vector_store.add(ids, texts, embeddings, metadata)
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

    def reindex_file(self, path: Path) -> IndexResult:
        """Re-indexiert eine Datei (Delta Update).

        Loescht zuerst alle alten Daten der Datei aus Graph und
        Vector Store und indexiert sie dann neu.

        Args:
            path: Pfad zur zu re-indexierenden Datei.

        Returns:
            IndexResult mit Statistiken und ggf. Fehlern.
        """
        start = time.monotonic()
        path_str = str(path)

        # Cleanup-Phase: alte Daten loeschen
        try:
            self._notify("cleanup", f"deleting old data for {path_str}")
            chunk_ids = self.graph.get_chunk_ids_for_file(path_str)
            self.graph.delete_nodes_for_file(path_str)
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
                errors=[f"cleanup failed for {path_str}: {e}"],
            )

        # Neu-Indexierung
        return self.index_file(path)
