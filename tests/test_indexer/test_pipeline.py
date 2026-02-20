"""Tests fuer nemesis.indexer.pipeline — Single File Index, Reindex, Full & Delta."""

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
    file: str = ""


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


# ---------------------------------------------------------------------------
# Task 9: index_project — Full Project Index
# ---------------------------------------------------------------------------


class TestIndexProject:
    def test_index_project_indexes_all_files(self, tmp_path: Path):
        """index_project indexiert alle passenden Dateien, ignoriert andere."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        (tmp_path / "readme.md").write_text("# Readme\n")

        node = MockCodeNode(
            id="v-1",
            node_type="variable",
            name="x",
            start_line=1,
            end_line=1,
            source="x = 1\n",
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

        result = pipeline.index_project(tmp_path, languages=["python"])

        # Nur die 2 .py Dateien, nicht die .md
        assert parser.parse_file.call_count == 2
        assert result.files_indexed == 2

    def test_index_project_skips_ignored_dirs(self, tmp_path: Path):
        """__pycache__/ Dateien werden uebersprungen."""
        (tmp_path / "main.py").write_text("a = 1\n")
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("b = 2\n")

        node = MockCodeNode(
            id="v-a",
            node_type="variable",
            name="a",
            start_line=1,
            end_line=1,
            source="a = 1\n",
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

        result = pipeline.index_project(tmp_path, languages=["python"])

        # Nur main.py, nicht __pycache__/cached.py
        assert parser.parse_file.call_count == 1
        assert result.files_indexed == 1

    def test_index_project_multiple_languages(self, tmp_path: Path):
        """index_project findet Dateien mehrerer Sprachen."""
        (tmp_path / "app.py").write_text("x = 1\n")
        (tmp_path / "index.ts").write_text("const x = 1;\n")

        node = MockCodeNode(
            id="v-multi",
            node_type="variable",
            name="x",
            start_line=1,
            end_line=1,
            source="x = 1\n",
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

        result = pipeline.index_project(tmp_path, languages=["python", "typescript"])

        assert parser.parse_file.call_count == 2
        assert result.files_indexed == 2

    def test_index_project_empty_dir(self, tmp_path: Path):
        """Leeres Verzeichnis ergibt 0 Dateien, success=True."""
        parser = _make_mock_parser()
        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.index_project(tmp_path, languages=["python"])

        assert result.files_indexed == 0
        assert result.success is True
        parser.parse_file.assert_not_called()

    def test_index_project_continues_on_single_file_error(self, tmp_path: Path):
        """Fehler bei einer Datei unterbricht nicht die anderen."""
        (tmp_path / "good.py").write_text("x = 1\n")
        (tmp_path / "bad.py").write_text("broken!\n")

        node = MockCodeNode(
            id="v-good",
            node_type="variable",
            name="x",
            start_line=1,
            end_line=1,
            source="x = 1\n",
        )
        good_result = MockParseResult(nodes=[node], edges=[])

        # Parser: erste Datei (bad.py, alphabetisch) wirft Fehler, zweite (good.py) OK
        parser = MagicMock()

        def _parse_side_effect(path_str):
            if "bad.py" in path_str:
                raise RuntimeError("Parse error")
            return good_result

        parser.parse_file.side_effect = _parse_side_effect

        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.index_project(tmp_path, languages=["python"])

        # Beide Dateien wurden verarbeitet
        assert parser.parse_file.call_count == 2
        # Mindestens eine erfolgreich
        assert result.files_indexed >= 1
        # Fehler von bad.py gesammelt
        assert len(result.errors) >= 1

    def test_index_project_progress_reporting(self, tmp_path: Path):
        """on_progress Callback wird fuer jede Datei aufgerufen."""
        (tmp_path / "one.py").write_text("a = 1\n")
        (tmp_path / "two.py").write_text("b = 2\n")

        node = MockCodeNode(
            id="v-prog",
            node_type="variable",
            name="a",
            start_line=1,
            end_line=1,
            source="a = 1\n",
        )
        parse_result = MockParseResult(nodes=[node], edges=[])

        parser = _make_mock_parser(parse_result)
        graph = _make_mock_graph()
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        progress_calls: list[tuple[str, str]] = []

        def _on_progress(step: str, detail: str) -> None:
            progress_calls.append((step, detail))

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
            on_progress=_on_progress,
        )

        pipeline.index_project(tmp_path, languages=["python"])

        # Mindestens 2 index_project Aufrufe (einen pro Datei)
        project_calls = [c for c in progress_calls if c[0] == "index_project"]
        assert len(project_calls) >= 2


# ---------------------------------------------------------------------------
# Task 10: update_project — Delta Project Update
# ---------------------------------------------------------------------------


class TestUpdateProject:
    def test_update_project_processes_all_change_types(self, tmp_path: Path):
        """Modified + neue Dateien auf Disk, geloeschte im Graph."""
        (tmp_path / "new.py").write_text("x = 1\n")
        (tmp_path / "modified.py").write_text("y = 2\n")

        node = MockCodeNode(
            id="v-up",
            node_type="variable",
            name="x",
            start_line=1,
            end_line=1,
            source="x = 1\n",
        )
        parse_result = MockParseResult(nodes=[node], edges=[])

        parser = _make_mock_parser(parse_result)
        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {
            str(tmp_path / "modified.py"): "old-hash",
            str(tmp_path / "deleted.py"): "deleted-hash",
        }
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.update_project(tmp_path, languages=["python"])

        # Alle 3 Change-Typen verarbeitet
        assert result.files_indexed >= 3
        # Parser aufgerufen fuer new.py (index_file) + modified.py (reindex_file)
        assert parser.parse_file.call_count >= 2

    def test_update_project_no_changes(self, tmp_path: Path):
        """Keine Aenderungen -> 0 indexiert, Parser nicht aufgerufen."""
        (tmp_path / "stable.py").write_text("x = 1\n")

        # Hash berechnen, damit detect_changes "unchanged" erkennt
        from nemesis.indexer.delta import compute_file_hash

        real_hash = compute_file_hash(tmp_path / "stable.py")

        parser = _make_mock_parser()
        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {
            str(tmp_path / "stable.py"): real_hash,
        }
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.update_project(tmp_path, languages=["python"])

        assert result.files_indexed == 0
        parser.parse_file.assert_not_called()

    def test_update_project_handles_deleted_files(self, tmp_path: Path):
        """Datei im Graph, aber nicht auf Disk -> delete_nodes_for_file."""
        # Leeres Verzeichnis — keine Dateien auf Disk
        parser = _make_mock_parser()
        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {
            str(tmp_path / "gone.py"): "old-hash",
        }
        vector_store = _make_mock_vector_store()
        embedder = _make_mock_embedder()

        pipeline = IndexingPipeline(
            parser=parser,
            graph=graph,
            vector_store=vector_store,
            embedder=embedder,
        )

        result = pipeline.update_project(tmp_path, languages=["python"])

        # delete_nodes_for_file wurde aufgerufen
        graph.delete_nodes_for_file.assert_called_once_with(str(tmp_path / "gone.py"))
        assert result.files_indexed >= 1
        # Parser nicht aufgerufen (nur Loeschung)
        parser.parse_file.assert_not_called()


# ---------------------------------------------------------------------------
# Task 4 (Multi-Project): project_id Prefix und relative Pfade
# ---------------------------------------------------------------------------


class TestPipelineProjectId:
    def test_index_file_prefixes_node_ids(self, tmp_path: Path):
        """Node IDs get project_id:: prefix."""
        source_code = "def hello():\n    return 42\n"
        test_file = tmp_path / "hello.py"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="func:hello.py:hello:1",
            node_type="function",
            name="hello",
            start_line=1,
            end_line=2,
            source=source_code,
        )
        edge = MockEdge(
            source_id="file:hello.py",
            target_id="func:hello.py:hello:1",
            kind="CONTAINS",
        )
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
        result = pipeline.index_file(test_file, project_id="eve", project_root=tmp_path)

        # Check node IDs are prefixed
        node_call = graph.add_node.call_args[0][0]
        assert node_call.id.startswith("eve::"), f"Node ID missing prefix: {node_call.id}"

        # Check edge IDs are prefixed
        edge_call = graph.add_edge.call_args[0][0]
        assert edge_call.source_id.startswith("eve::"), (
            f"Edge source missing prefix: {edge_call.source_id}"
        )
        assert edge_call.target_id.startswith("eve::"), (
            f"Edge target missing prefix: {edge_call.target_id}"
        )

    def test_index_file_uses_relative_paths(self, tmp_path: Path):
        """File paths stored in graph are relative to project_root."""
        sub = tmp_path / "services"
        sub.mkdir()
        test_file = sub / "main.py"
        source_code = "x = 1\n"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="var:main.py:x:1",
            node_type="variable",
            name="x",
            start_line=1,
            end_line=1,
            source=source_code,
        )
        # The node.file field is set by the parser to the absolute path
        node.file = str(test_file)
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
        result = pipeline.index_file(test_file, project_id="eve", project_root=tmp_path)

        node_call = graph.add_node.call_args[0][0]
        # MockCodeNode wird direkt durchgereicht — pruefe node.file
        assert not node_call.file.startswith("/"), f"Path should be relative: {node_call.file}"
        assert node_call.file == "services/main.py"

    def test_index_file_passes_project_id_to_vector_store(self, tmp_path: Path):
        """project_id is passed to vector_store.add()."""
        source_code = "def fn():\n    pass\n"
        test_file = tmp_path / "fn.py"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="func:fn.py:fn:1",
            node_type="function",
            name="fn",
            start_line=1,
            end_line=2,
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
        pipeline.index_file(test_file, project_id="eve", project_root=tmp_path)

        # vector_store.add was called with project_id
        vector_store.add.assert_called_once()
        call_kwargs = vector_store.add.call_args
        # Could be positional or keyword — check both
        assert call_kwargs.kwargs.get("project_id") == "eve" or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] == "eve"
        )

    def test_index_file_without_project_id_works(self, tmp_path: Path):
        """Backward compatible — no project_id means no prefix."""
        source_code = "def hello():\n    return 42\n"
        test_file = tmp_path / "hello.py"
        test_file.write_text(source_code)

        node = MockCodeNode(
            id="func:hello.py:hello:1",
            node_type="function",
            name="hello",
            start_line=1,
            end_line=2,
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
        result = pipeline.index_file(test_file)

        node_call = graph.add_node.call_args[0][0]
        assert not node_call.id.startswith("::"), "No prefix when project_id is empty"

    def test_index_project_passes_project_id(self, tmp_path: Path):
        """index_project passes project_id and project_root to index_file."""
        (tmp_path / "a.py").write_text("x = 1\n")

        node = MockCodeNode(
            id="v-1",
            node_type="variable",
            name="x",
            start_line=1,
            end_line=1,
            source="x = 1\n",
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
        result = pipeline.index_project(
            tmp_path, languages=["python"], project_id="myproj", project_root=tmp_path,
        )

        # Node should have prefix
        node_call = graph.add_node.call_args[0][0]
        assert node_call.id.startswith("myproj::"), f"Node ID missing prefix: {node_call.id}"
