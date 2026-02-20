# 05d — Single File Index + Reindex

> **Arbeitspaket E4** — Teil 4 von 5 des Indexing Pipeline Plans

**Goal:** Die zentrale IndexingPipeline-Klasse implementieren: Einzelne Dateien indexieren (Parse -> Chunk -> Embed -> Store) und inkrementell re-indexieren (alte Daten loeschen, dann neu indexieren).

**Tech Stack:** Python, nemesis.indexer.chunker, nemesis.indexer.delta, nemesis.indexer.models

**Abhängigkeiten:** E1 (05a), E2 (05b), E3 (05c)

**Tasks in diesem Paket:** 7, 8

---

## Task 7: Pipeline — Single File Index

**Files:**
- `nemesis/indexer/pipeline.py`
- `tests/test_indexer/test_pipeline.py`

### Step 1 — Write Test

```python
# tests/test_indexer/test_pipeline.py
"""Tests fuer die Indexing Pipeline."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockParseResult:
    """Mock fuer das Parser-Ergebnis."""

    nodes: list
    edges: list
    file_path: str
    language: str


@dataclass
class MockCodeNode:
    """Mock fuer einen Code-Knoten."""

    id: str
    node_type: str
    name: str
    start_line: int
    end_line: int
    source: str
    children: list = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class MockEdge:
    """Mock fuer eine Graph-Kante."""

    source_id: str
    target_id: str
    edge_type: str


def _make_mock_parser():
    """Erzeugt einen Mock-Parser."""
    parser = MagicMock()
    node = MockCodeNode(
        id="func-001",
        node_type="Function",
        name="hello",
        start_line=1,
        end_line=3,
        source="def hello():\n    return 42\n",
    )
    edge = MockEdge(
        source_id="file-main",
        target_id="func-001",
        edge_type="CONTAINS",
    )
    result = MockParseResult(
        nodes=[node],
        edges=[edge],
        file_path="src/main.py",
        language="python",
    )
    parser.parse_file.return_value = result
    return parser, result


def _make_mock_graph():
    graph = MagicMock()
    graph.add_node.return_value = None
    graph.add_edge.return_value = None
    graph.get_chunk_ids_for_file.return_value = []
    graph.delete_nodes_for_file.return_value = None
    graph.get_file_hashes.return_value = {}
    return graph


def _make_mock_vector_store():
    store = MagicMock()
    store.add_embeddings.return_value = None
    store.delete_embeddings.return_value = None
    return store


def _make_mock_embedder():
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    return embedder


def test_pipeline_creation():
    """Pipeline laesst sich mit allen Abhaengigkeiten erstellen."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser = MagicMock()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )
    assert pipeline is not None


def test_index_file_parses_and_stores():
    """index_file parsed die Datei und speichert Nodes/Edges/Embeddings."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser, parse_result = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_file(Path("src/main.py"))

    # Parser wurde aufgerufen
    parser.parse_file.assert_called_once_with(Path("src/main.py"))
    # Nodes im Graph gespeichert
    assert graph.add_node.called
    # Edges im Graph gespeichert
    assert graph.add_edge.called
    # Embeddings erzeugt und gespeichert
    assert embedder.embed_texts.called
    assert vector_store.add_embeddings.called
    # Ergebnis korrekt
    assert result.files_indexed == 1
    assert result.nodes_created >= 1
    assert result.edges_created >= 1
    assert result.success is True


def test_index_file_returns_index_result():
    """index_file gibt ein korrektes IndexResult zurueck."""
    from nemesis.indexer.pipeline import IndexingPipeline
    from nemesis.indexer.models import IndexResult

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_file(Path("src/main.py"))

    assert isinstance(result, IndexResult)
    assert result.duration_ms >= 0
    assert result.chunks_created >= 0
    assert result.embeddings_created >= 0


def test_index_file_handles_parser_error():
    """index_file faengt Parser-Fehler ab und meldet sie im Ergebnis."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser = MagicMock()
    parser.parse_file.side_effect = RuntimeError("Parse error: invalid syntax")
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_file(Path("broken.py"))

    assert result.success is False
    assert len(result.errors) == 1
    assert "broken.py" in result.errors[0]


def test_index_file_chunks_large_nodes():
    """Grosse Nodes werden in Chunks aufgeteilt."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser = MagicMock()
    # Erzeuge einen grossen Node
    big_source = "def big():\n" + "\n".join(f"    x_{i} = {i}" for i in range(200)) + "\n"
    big_node = MockCodeNode(
        id="func-big",
        node_type="Function",
        name="big",
        start_line=1,
        end_line=201,
        source=big_source,
    )
    parse_result = MockParseResult(
        nodes=[big_node],
        edges=[],
        file_path="src/big.py",
        language="python",
    )
    parser.parse_file.return_value = parse_result

    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()
    embedder.embed_texts.return_value = [[0.1] * 3] * 20  # Genug fuer alle Chunks

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_file(Path("src/big.py"))

    assert result.chunks_created > 1
    assert result.success is True
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: FAILED — Module nemesis.indexer.pipeline existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/pipeline.py
"""Indexing Pipeline — Orchestriert Parse -> Chunk -> Embed -> Store.

Die zentrale Klasse IndexingPipeline bringt Parser, Graph-Adapter,
Vector Store und Embedder zusammen. Sie unterstuetzt:
- Full Index eines gesamten Projekts
- Single File Index
- Delta Update (delete old + insert new)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from nemesis.indexer.chunker import chunk_node
from nemesis.indexer.models import Chunk, IndexResult

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Orchestriert den Indexierungs-Datenfluss.

    Verbindet Parser (Rust/Tree-sitter), AST-Chunker,
    Embedding-Generator und Graph/Vector Stores.
    """

    def __init__(
        self,
        parser: Any,
        graph: Any,
        vector_store: Any,
        embedder: Any,
        max_tokens_per_chunk: int = 500,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Erstelle eine neue IndexingPipeline.

        Args:
            parser: Parser mit parse_file(path) -> ParseResult.
            graph: GraphAdapter mit add_node(), add_edge(), etc.
            vector_store: VectorStore mit add_embeddings(), etc.
            embedder: Embedder mit embed_texts(texts) -> list[list[float]].
            max_tokens_per_chunk: Maximale Token-Anzahl pro Chunk.
            on_progress: Optionaler Callback (message, current, total).
        """
        self._parser = parser
        self._graph = graph
        self._vector_store = vector_store
        self._embedder = embedder
        self._max_tokens = max_tokens_per_chunk
        self._on_progress = on_progress

    def _report_progress(self, message: str, current: int = 0, total: int = 0) -> None:
        """Melde Fortschritt ueber den Callback."""
        if self._on_progress:
            self._on_progress(message, current, total)
        logger.debug("Progress: %s (%d/%d)", message, current, total)

    def index_file(self, path: Path) -> IndexResult:
        """Indexiere eine einzelne Datei.

        Ablauf:
        1. Parse die Datei -> Nodes + Edges
        2. Chunke grosse Nodes
        3. Erzeuge Embeddings fuer alle Chunks
        4. Speichere Nodes + Edges im Graph
        5. Speichere Embeddings im Vector Store

        Args:
            path: Pfad zur Datei.

        Returns:
            IndexResult mit Statistiken.
        """
        start_time = time.monotonic()
        nodes_created = 0
        edges_created = 0
        chunks_created = 0
        embeddings_created = 0
        errors: list[str] = []

        try:
            # 1. Parse
            self._report_progress(f"Parsing {path.name}")
            parse_result = self._parser.parse_file(path)

            # 2. Nodes im Graph speichern
            for node in parse_result.nodes:
                self._graph.add_node(node)
                nodes_created += 1

            # 3. Edges im Graph speichern
            for edge in parse_result.edges:
                self._graph.add_edge(edge)
                edges_created += 1

            # 4. Chunking
            self._report_progress(f"Chunking {path.name}")
            all_chunks: list[Chunk] = []
            for node in parse_result.nodes:
                source = getattr(node, "source", "")
                if source:
                    node_chunks = chunk_node(
                        node,
                        source,
                        max_tokens=self._max_tokens,
                        file_path=path,
                    )
                    all_chunks.extend(node_chunks)

            chunks_created = len(all_chunks)

            # 5. Embeddings erzeugen
            if all_chunks:
                self._report_progress(f"Embedding {len(all_chunks)} chunks")
                texts = [chunk.content for chunk in all_chunks]
                vectors = self._embedder.embed_texts(texts)

                # 6. Im Vector Store speichern
                self._report_progress(f"Storing embeddings")
                embedding_data = []
                for chunk, vector in zip(all_chunks, vectors):
                    embedding_data.append(
                        {
                            "id": chunk.id,
                            "vector": vector,
                            "content": chunk.content,
                            "metadata": {
                                "file_path": str(chunk.file_path),
                                "parent_node_id": chunk.parent_node_id,
                                "parent_type": chunk.parent_type,
                                "start_line": chunk.start_line,
                                "end_line": chunk.end_line,
                            },
                        }
                    )
                self._vector_store.add_embeddings(embedding_data)
                embeddings_created = len(embedding_data)

                # Chunk-Nodes auch im Graph speichern
                for chunk in all_chunks:
                    self._graph.add_node(chunk)

        except Exception as e:
            errors.append(f"Error indexing {path}: {e}")
            logger.error("Failed to index %s: %s", path, e)

        duration_ms = (time.monotonic() - start_time) * 1000

        return IndexResult(
            files_indexed=1 if not errors else 0,
            nodes_created=nodes_created,
            edges_created=edges_created,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            duration_ms=duration_ms,
            errors=errors,
        )
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/pipeline.py tests/test_indexer/test_pipeline.py
git commit -m "feat(indexer): add IndexingPipeline with single file indexing"
```

---

## Task 8: Pipeline — Reindex File (Delta Update)

**Files:**
- `nemesis/indexer/pipeline.py` (erweitern)
- `tests/test_indexer/test_pipeline.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_pipeline.py — APPEND folgende Tests

def test_reindex_file_deletes_old_data_first():
    """reindex_file loescht alte Daten bevor es neu indexiert."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    graph.get_chunk_ids_for_file.return_value = ["old-chunk-1", "old-chunk-2"]
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.reindex_file(Path("src/changed.py"))

    # Alte Daten wurden geloescht
    graph.delete_nodes_for_file.assert_called_once_with(str(Path("src/changed.py")))
    vector_store.delete_embeddings.assert_called_once_with(["old-chunk-1", "old-chunk-2"])
    # Dann wurde neu indexiert
    parser.parse_file.assert_called_once()
    assert result.success is True
    assert result.files_indexed == 1


def test_reindex_file_returns_result():
    """reindex_file gibt ein korrektes IndexResult zurueck."""
    from nemesis.indexer.pipeline import IndexingPipeline
    from nemesis.indexer.models import IndexResult

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.reindex_file(Path("src/changed.py"))

    assert isinstance(result, IndexResult)
    assert result.duration_ms >= 0


def test_reindex_preserves_other_files():
    """reindex_file loescht nur Daten der angegebenen Datei."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    pipeline.reindex_file(Path("src/file_a.py"))

    # Nur file_a.py wurde geloescht, nicht andere Dateien
    call_args = graph.delete_nodes_for_file.call_args[0][0]
    assert "file_a.py" in call_args


def test_reindex_file_handles_delete_error():
    """reindex_file meldet Fehler wenn Loeschen fehlschlaegt."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    graph.get_chunk_ids_for_file.side_effect = RuntimeError("DB connection lost")
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.reindex_file(Path("src/broken.py"))

    assert result.success is False
    assert len(result.errors) >= 1
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v -k "reindex"
# Erwartung: FAILED — reindex_file existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/pipeline.py — Methode reindex_file zur Klasse IndexingPipeline hinzufuegen

    def reindex_file(self, path: Path) -> IndexResult:
        """Inkrementeller Re-Index einer geaenderten Datei.

        Ablauf:
        1. Alte Chunks-IDs aus dem Graph laden
        2. Alte Nodes/Edges im Graph loeschen
        3. Alte Embeddings im Vector Store loeschen
        4. Datei neu indexieren (wie index_file)

        Dies ist der Delta-Update-Pfad: Nur die geaenderte Datei
        wird neu verarbeitet, nicht das gesamte Projekt.

        Args:
            path: Pfad zur geaenderten Datei.

        Returns:
            IndexResult mit Statistiken.
        """
        start_time = time.monotonic()

        try:
            # 1. Alte Daten loeschen
            self._report_progress(f"Cleaning old data for {path.name}")
            chunk_ids = self._graph.get_chunk_ids_for_file(str(path))
            self._graph.delete_nodes_for_file(str(path))
            self._vector_store.delete_embeddings(chunk_ids)
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            return IndexResult(
                files_indexed=0,
                nodes_created=0,
                edges_created=0,
                chunks_created=0,
                embeddings_created=0,
                duration_ms=duration_ms,
                errors=[f"Error cleaning old data for {path}: {e}"],
            )

        # 2. Neu indexieren
        result = self.index_file(path)

        # Duration korrigieren (inkl. Loeschzeit)
        total_duration_ms = (time.monotonic() - start_time) * 1000
        result.duration_ms = total_duration_ms

        return result
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/pipeline.py tests/test_indexer/test_pipeline.py
git commit -m "feat(indexer): add reindex_file for incremental delta updates"
```

---

## Zusammenfassung E4

| Task | Datei(en) | Beschreibung |
|------|-----------|-------------|
| 7 | `pipeline.py` | IndexingPipeline: Single File Index |
| 8 | `pipeline.py` | reindex_file: Delta Update fuer einzelne Dateien |

---

**Navigation:**
- Zurueck: [05c — Delta Operations (E3)](05c-delta-ops.md)
- Weiter: [05e — Full Project Pipeline (E5)](05e-full-project-pipeline.md)
