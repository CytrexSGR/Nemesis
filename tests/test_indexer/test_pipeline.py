"""Tests fuer nemesis.indexer.pipeline — Single File Index + Reindex."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from nemesis.indexer.pipeline import IndexingPipeline

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Mock Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockCodeNode:
    """Mock fuer einen Code-Knoten mit den Feldern die chunk_node erwartet."""

    id: str
    node_type: str
    name: str
    start_line: int
    end_line: int
    source: str = ""


@dataclass
class MockEdge:
    """Mock fuer eine Code-Kante."""

    source_id: str
    target_id: str
    kind: str


@dataclass
class MockParseResult:
    """Mock fuer das Ergebnis von parser.parse_file()."""

    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)


def _make_mock_parser(parse_result=None, error=None):
    """Erstellt einen Mock-Parser."""
    parser = MagicMock()
    if error is not None:
        parser.parse_file.side_effect = error
    else:
        parser.parse_file.return_value = parse_result or MockParseResult()
    return parser


def _make_mock_graph():
    """Erstellt einen Mock-Graph-Adapter."""
    graph = MagicMock()
    graph.get_chunk_ids_for_file.return_value = []
    return graph


def _make_mock_vector_store():
    """Erstellt einen Mock-VectorStore."""
    return MagicMock()


def _make_mock_embedder(dimensions=384):
    """Erstellt einen Mock-EmbeddingProvider."""
    embedder = MagicMock()

    def _mock_embed(texts):
        result = MagicMock()
        result.embeddings = [[0.1] * dimensions for _ in texts]
        result.total_tokens = len(texts) * 10
        return result

    embedder.embed.side_effect = _mock_embed
    embedder.dimensions = dimensions
    return embedder


# ---------------------------------------------------------------------------
# Task 7: IndexingPipeline — Single File Index
# ---------------------------------------------------------------------------


class TestPipelineCreation:
    def test_pipeline_creation(self):
        """Pipeline laesst sich mit allen Abhaengigkeiten erstellen."""
        parser = _make_mock_parser()
        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
            max_tokens_per_chunk=500,
            on_progress=None,
        )

        assert pipeline.parser is parser
        assert pipeline.graph is graph
        assert pipeline.vector_store is vector_store
        assert pipeline.embedder is embedder
        assert pipeline.max_tokens_per_chunk == 500


class TestIndexFile:
    def test_index_file_parses_and_stores(self, tmp_path: Path):
        """index_file ruft parser, graph.add_node, graph.add_edge, embedder, vector_store auf."""
        source_code = "def hello():\n    return 42\n"
        test_file = tmp_path / "hello.py"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="func-1",
            node_type="function",
            name="hello",
            start_line=1,
            end_line=2,
            source=source_code,
        )
        edge = MockEdge(source_id="file-1", target_id="func-1", kind="CONTAINS")
        parse_result = MockParseResult(nodes=[node], edges=[edge])

        parser = _make_mock_parser(parse_result)
        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.index_file(test_file)

        # Parser wurde aufgerufen
        parser.parse_file.assert_called_once_with(str(test_file))

        # Graph: add_node und add_edge wurden aufgerufen
        graph.add_node.assert_called_once_with(node)
        graph.add_edge.assert_called_once_with(edge)

        # Embedder wurde aufgerufen
        embedder.embed.assert_called_once()

        # Vector Store: add wurde aufgerufen
        vector_store.add.assert_called_once()

        assert result.success

    def test_index_file_returns_index_result(self, tmp_path: Path):
        """index_file gibt ein IndexResult mit korrekten Feldern zurueck."""
        source_code = "def greet(name):\n    print(f'Hello {name}')\n"
        test_file = tmp_path / "greet.py"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="func-2",
            node_type="function",
            name="greet",
            start_line=1,
            end_line=2,
            source=source_code,
        )
        edge = MockEdge(source_id="file-2", target_id="func-2", kind="CONTAINS")
        parse_result = MockParseResult(nodes=[node], edges=[edge])

        parser = _make_mock_parser(parse_result)
        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.index_file(test_file)

        assert result.success is True
        assert result.nodes_created == 1
        assert result.edges_created == 1
        assert result.chunks_created >= 1
        assert result.embeddings_created >= 1
        assert result.duration_ms >= 0
        assert result.errors == []

    def test_index_file_handles_parser_error(self, tmp_path: Path):
        """Bei RuntimeError vom Parser wird success=False zurueckgegeben."""
        test_file = tmp_path / "broken.py"
        test_file.write_text("syntax error!!!\n")

        parser = _make_mock_parser(error=RuntimeError("Parse failed"))
        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.index_file(test_file)

        assert result.success is False
        assert len(result.errors) >= 1
        assert "Parse failed" in result.errors[0]

    def test_index_file_chunks_large_nodes(self, tmp_path: Path):
        """Grosser Knoten (200 Zeilen) wird in mehrere Chunks aufgeteilt."""
        # Erstelle Source-Code mit 200 Zeilen
        lines = ["class BigClass:"]
        for i in range(200):
            lines.append(f"    def method_{i}(self):")
            lines.append(f"        x = {i} * 2")
            lines.append(f"        return x + {i}")
            lines.append("")
        source_code = "\n".join(lines) + "\n"

        test_file = tmp_path / "big.py"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="cls-big",
            node_type="class",
            name="BigClass",
            start_line=1,
            end_line=len(lines),
            source=source_code,
        )
        parse_result = MockParseResult(nodes=[node], edges=[])

        parser = _make_mock_parser(parse_result)
        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
            max_tokens_per_chunk=500,
        )

        result = pipeline.index_file(test_file)

        assert result.success is True
        # 200 Zeilen Source sollte in mehrere Chunks aufgeteilt werden
        assert result.chunks_created > 1
        assert result.embeddings_created == result.chunks_created


# ---------------------------------------------------------------------------
# Task 8: reindex_file (Delta Update)
# ---------------------------------------------------------------------------


class TestReindexFile:
    def test_reindex_file_deletes_old_data_first(self, tmp_path: Path):
        """reindex_file ruft get_chunk_ids_for_file, delete_nodes_for_file,
        delete_embeddings auf bevor neu indexiert wird."""
        source_code = "def hello():\n    return 42\n"
        test_file = tmp_path / "hello.py"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="func-1",
            node_type="function",
            name="hello",
            start_line=1,
            end_line=2,
            source=source_code,
        )
        parse_result = MockParseResult(nodes=[node], edges=[])

        parser = _make_mock_parser(parse_result)
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

        pipeline.reindex_file(test_file)

        # Alte Daten wurden abgefragt und geloescht
        graph.get_chunk_ids_for_file.assert_called_once_with(str(test_file))
        graph.delete_nodes_for_file.assert_called_once_with(str(test_file))
        vector_store.delete_embeddings.assert_called_once_with(["old-chunk-1", "old-chunk-2"])

    def test_reindex_file_returns_result(self, tmp_path: Path):
        """reindex_file gibt ein IndexResult zurueck."""
        source_code = "x = 1\n"
        test_file = tmp_path / "simple.py"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="var-1",
            node_type="variable",
            name="x",
            start_line=1,
            end_line=1,
            source=source_code,
        )
        parse_result = MockParseResult(nodes=[node], edges=[])

        parser = _make_mock_parser(parse_result)
        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.reindex_file(test_file)

        assert result is not None
        assert result.success is True
        assert result.nodes_created >= 1

    def test_reindex_preserves_other_files(self, tmp_path: Path):
        """reindex_file loescht nur die Daten der Ziel-Datei."""
        source_code = "y = 2\n"
        target_file = tmp_path / "target.py"
        target_file.write_text(source_code)

        node = MockCodeNode(
            id="var-2",
            node_type="variable",
            name="y",
            start_line=1,
            end_line=1,
            source=source_code,
        )
        parse_result = MockParseResult(nodes=[node], edges=[])

        parser = _make_mock_parser(parse_result)
        graph = _make_mock_graph()
        graph.get_chunk_ids_for_file.return_value = ["chunk-target"]
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        pipeline.reindex_file(target_file)

        # Nur target_file Daten loeschen — nicht andere Dateien
        graph.get_chunk_ids_for_file.assert_called_once_with(str(target_file))
        graph.delete_nodes_for_file.assert_called_once_with(str(target_file))
        vector_store.delete_embeddings.assert_called_once_with(["chunk-target"])

    def test_reindex_file_handles_delete_error(self, tmp_path: Path):
        """Bei Fehler waehrend Cleanup wird success=False zurueckgegeben."""
        test_file = tmp_path / "error.py"
        test_file.write_text("z = 3\n")

        parser = _make_mock_parser()
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

        result = pipeline.reindex_file(test_file)

        assert result.success is False
        assert len(result.errors) >= 1
        assert "cleanup failed" in result.errors[0]
