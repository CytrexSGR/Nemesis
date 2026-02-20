"""Tests for Kuzu graph adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemesis.graph.adapter import (
    NODE_TYPES,
    EdgeData,
    GraphAdapter,
    NodeData,
    TraversalResult,
)
from nemesis.graph.kuzu import KuzuAdapter

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class TestKuzuInit:
    """Tests for KuzuAdapter initialization and schema."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "test_graph")

    @pytest.fixture
    def adapter(self, db_path: str) -> Generator[KuzuAdapter, None, None]:
        a = KuzuAdapter(db_path=db_path)
        a.create_schema()
        yield a
        a.close()

    def test_is_graph_adapter(self, adapter: KuzuAdapter) -> None:
        assert isinstance(adapter, GraphAdapter)

    def test_create_schema_idempotent(self, db_path: str) -> None:
        """Calling create_schema twice should not raise."""
        a = KuzuAdapter(db_path=db_path)
        a.create_schema()
        a.create_schema()  # Should not raise
        a.close()

    def test_schema_creates_all_node_tables(self, adapter: KuzuAdapter) -> None:
        """All node types from the design doc have a table."""
        for node_type in NODE_TYPES:
            # Verify by inserting a minimal node -- should not raise
            node = NodeData(id=f"test-{node_type}", node_type=node_type)
            adapter.add_node(node)
            result = adapter.get_node(f"test-{node_type}")
            assert result is not None, f"Node table for {node_type} not found"

    def test_schema_creates_all_edge_tables(self, adapter: KuzuAdapter) -> None:
        """All edge types from the design doc have a relationship table."""
        # We verify edge tables exist by checking schema -- they're created
        # even if empty. Add source and target nodes first.
        adapter.add_node(NodeData(id="src-node", node_type="File", properties={"path": "a.py"}))
        adapter.add_node(NodeData(id="tgt-node", node_type="Function", properties={"name": "f"}))
        # CONTAINS is a valid edge between File and Function
        edge = EdgeData(source_id="src-node", target_id="tgt-node", edge_type="CONTAINS")
        adapter.add_edge(edge)  # Should not raise

    def test_close_and_reopen(self, db_path: str) -> None:
        """Data persists after close and reopen."""
        a1 = KuzuAdapter(db_path=db_path)
        a1.create_schema()
        a1.add_node(NodeData(id="persistent", node_type="File", properties={"path": "test.py"}))
        a1.close()

        a2 = KuzuAdapter(db_path=db_path)
        a2.create_schema()
        node = a2.get_node("persistent")
        assert node is not None
        assert node.id == "persistent"
        a2.close()


class TestKuzuNodeCRUD:
    """Tests for node add/get/delete operations."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> Generator[KuzuAdapter, None, None]:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    def test_add_and_get_file_node(self, adapter: KuzuAdapter) -> None:
        node = NodeData(
            id="file-main",
            node_type="File",
            properties={
                "path": "/src/main.py",
                "language": "python",
                "hash": "abc123",
                "size": 1024,
            },
        )
        adapter.add_node(node)
        result = adapter.get_node("file-main")
        assert result is not None
        assert result.id == "file-main"
        assert result.node_type == "File"
        assert result.properties["path"] == "/src/main.py"
        assert result.properties["language"] == "python"

    def test_add_and_get_function_node(self, adapter: KuzuAdapter) -> None:
        node = NodeData(
            id="func-hello",
            node_type="Function",
            properties={
                "name": "hello",
                "file": "main.py",
                "line_start": 10,
                "line_end": 15,
                "signature": "def hello(name: str) -> str",
                "docstring": "Greet someone.",
                "is_async": False,
            },
        )
        adapter.add_node(node)
        result = adapter.get_node("func-hello")
        assert result is not None
        assert result.properties["name"] == "hello"
        assert result.properties["line_start"] == 10
        assert result.properties["is_async"] is False

    def test_add_and_get_class_node(self, adapter: KuzuAdapter) -> None:
        node = NodeData(
            id="class-calc",
            node_type="Class",
            properties={
                "name": "Calculator",
                "file": "calc.py",
                "line_start": 1,
                "line_end": 50,
                "docstring": "A calculator.",
            },
        )
        adapter.add_node(node)
        result = adapter.get_node("class-calc")
        assert result is not None
        assert result.properties["name"] == "Calculator"

    def test_add_and_get_chunk_node(self, adapter: KuzuAdapter) -> None:
        node = NodeData(
            id="chunk-001",
            node_type="Chunk",
            properties={
                "content": "def hello(): pass",
                "token_count": 5,
                "embedding_id": "emb-001",
                "parent_type": "Function",
            },
        )
        adapter.add_node(node)
        result = adapter.get_node("chunk-001")
        assert result is not None
        assert result.properties["token_count"] == 5

    def test_get_nonexistent_node_returns_none(self, adapter: KuzuAdapter) -> None:
        result = adapter.get_node("does-not-exist")
        assert result is None

    def test_add_node_with_missing_properties_uses_defaults(self, adapter: KuzuAdapter) -> None:
        """Missing properties get default values (empty string, 0, false)."""
        node = NodeData(id="func-minimal", node_type="Function")
        adapter.add_node(node)
        result = adapter.get_node("func-minimal")
        assert result is not None
        assert result.properties["name"] == ""
        assert result.properties["line_start"] == 0

    def test_update_existing_node(self, adapter: KuzuAdapter) -> None:
        """Adding a node with same ID updates the properties."""
        adapter.add_node(
            NodeData(
                id="file-x",
                node_type="File",
                properties={"path": "x.py", "hash": "old_hash"},
            )
        )
        adapter.add_node(
            NodeData(
                id="file-x",
                node_type="File",
                properties={"path": "x.py", "hash": "new_hash"},
            )
        )
        result = adapter.get_node("file-x")
        assert result is not None
        assert result.properties["hash"] == "new_hash"

    def test_delete_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(id="del-me", node_type="Variable", properties={"name": "x"}))
        assert adapter.get_node("del-me") is not None
        adapter.delete_node("del-me")
        assert adapter.get_node("del-me") is None

    def test_delete_nonexistent_node_no_error(self, adapter: KuzuAdapter) -> None:
        """Deleting a non-existent node should not raise."""
        adapter.delete_node("ghost-node")  # Should not raise


class TestKuzuEdges:
    """Tests for edge add and neighbor query operations."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> Generator[KuzuAdapter, None, None]:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    @pytest.fixture
    def populated_adapter(self, adapter: KuzuAdapter) -> KuzuAdapter:
        """Create a small graph with CONTAINS and HAS_METHOD edges."""
        adapter.add_node(
            NodeData(
                id="file-main",
                node_type="File",
                properties={"path": "main.py", "language": "python"},
            )
        )
        adapter.add_node(
            NodeData(
                id="func-hello",
                node_type="Function",
                properties={"name": "hello", "file": "main.py"},
            )
        )
        adapter.add_node(
            NodeData(
                id="class-calc",
                node_type="Class",
                properties={"name": "Calculator", "file": "main.py"},
            )
        )
        adapter.add_node(
            NodeData(
                id="method-add",
                node_type="Method",
                properties={
                    "name": "add",
                    "class_name": "Calculator",
                    "file": "main.py",
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="func-helper",
                node_type="Function",
                properties={"name": "helper", "file": "utils.py"},
            )
        )

        adapter.add_edge(
            EdgeData(source_id="file-main", target_id="func-hello", edge_type="CONTAINS")
        )
        adapter.add_edge(
            EdgeData(source_id="file-main", target_id="class-calc", edge_type="CONTAINS")
        )
        adapter.add_edge(
            EdgeData(source_id="class-calc", target_id="method-add", edge_type="HAS_METHOD")
        )
        adapter.add_edge(
            EdgeData(source_id="func-hello", target_id="func-helper", edge_type="CALLS")
        )
        return adapter

    def test_add_edge_contains(self, populated_adapter: KuzuAdapter) -> None:
        """CONTAINS edge between File and Function is created."""
        neighbors = populated_adapter.get_neighbors(
            "file-main", edge_type="CONTAINS", direction="outgoing"
        )
        ids = {n.id for n in neighbors}
        assert "func-hello" in ids
        assert "class-calc" in ids

    def test_add_edge_has_method(self, populated_adapter: KuzuAdapter) -> None:
        """HAS_METHOD edge between Class and Method is created."""
        neighbors = populated_adapter.get_neighbors(
            "class-calc", edge_type="HAS_METHOD", direction="outgoing"
        )
        ids = {n.id for n in neighbors}
        assert "method-add" in ids

    def test_add_edge_calls(self, populated_adapter: KuzuAdapter) -> None:
        """CALLS edge between Functions is created."""
        neighbors = populated_adapter.get_neighbors(
            "func-hello", edge_type="CALLS", direction="outgoing"
        )
        ids = {n.id for n in neighbors}
        assert "func-helper" in ids

    def test_get_neighbors_outgoing(self, populated_adapter: KuzuAdapter) -> None:
        neighbors = populated_adapter.get_neighbors("file-main", direction="outgoing")
        assert len(neighbors) >= 2

    def test_get_neighbors_incoming(self, populated_adapter: KuzuAdapter) -> None:
        neighbors = populated_adapter.get_neighbors(
            "func-hello", edge_type="CONTAINS", direction="incoming"
        )
        ids = {n.id for n in neighbors}
        assert "file-main" in ids

    def test_get_neighbors_nonexistent_node(self, populated_adapter: KuzuAdapter) -> None:
        neighbors = populated_adapter.get_neighbors("ghost", direction="outgoing")
        assert neighbors == []

    def test_add_edge_missing_source_no_error(self, adapter: KuzuAdapter) -> None:
        """Adding an edge with missing source node should not raise."""
        adapter.add_node(NodeData(id="target-only", node_type="Function", properties={"name": "f"}))
        adapter.add_edge(EdgeData(source_id="missing", target_id="target-only", edge_type="CALLS"))
        # No crash, edge just not created


class TestKuzuFileOperations:
    """Tests for file-related operations used by the indexing pipeline."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> Generator[KuzuAdapter, None, None]:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    @pytest.fixture
    def indexed_adapter(self, adapter: KuzuAdapter) -> KuzuAdapter:
        """Simulate an indexed file with nodes, edges, and chunks."""
        adapter.add_node(
            NodeData(
                id="file-svc",
                node_type="File",
                properties={
                    "path": "/project/service.py",
                    "language": "python",
                    "hash": "aaa111",
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="func-process",
                node_type="Function",
                properties={
                    "name": "process",
                    "file": "/project/service.py",
                    "line_start": 5,
                    "line_end": 20,
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="class-svc",
                node_type="Class",
                properties={
                    "name": "Service",
                    "file": "/project/service.py",
                    "line_start": 1,
                    "line_end": 50,
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="chunk-001",
                node_type="Chunk",
                properties={
                    "content": "def process():",
                    "token_count": 5,
                    "parent_type": "Function",
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="chunk-002",
                node_type="Chunk",
                properties={
                    "content": "class Service:",
                    "token_count": 3,
                    "parent_type": "Class",
                },
            )
        )

        # File -> contains -> nodes
        adapter.add_edge(
            EdgeData(
                source_id="file-svc",
                target_id="func-process",
                edge_type="CONTAINS",
            )
        )
        adapter.add_edge(
            EdgeData(source_id="file-svc", target_id="class-svc", edge_type="CONTAINS")
        )
        # Chunk -> chunk_of -> node
        adapter.add_edge(
            EdgeData(
                source_id="chunk-001",
                target_id="func-process",
                edge_type="CHUNK_OF",
            )
        )
        adapter.add_edge(
            EdgeData(source_id="chunk-002", target_id="class-svc", edge_type="CHUNK_OF")
        )

        # Also add a node from another file to ensure isolation
        adapter.add_node(
            NodeData(
                id="file-other",
                node_type="File",
                properties={
                    "path": "/project/other.py",
                    "language": "python",
                    "hash": "bbb222",
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="func-other",
                node_type="Function",
                properties={"name": "other", "file": "/project/other.py"},
            )
        )
        adapter.add_edge(
            EdgeData(
                source_id="file-other",
                target_id="func-other",
                edge_type="CONTAINS",
            )
        )
        return adapter

    def test_get_file_hashes(self, indexed_adapter: KuzuAdapter) -> None:
        hashes = indexed_adapter.get_file_hashes()
        assert hashes["/project/service.py"] == "aaa111"
        assert hashes["/project/other.py"] == "bbb222"
        assert len(hashes) == 2

    def test_get_file_hashes_empty_graph(self, adapter: KuzuAdapter) -> None:
        hashes = adapter.get_file_hashes()
        assert hashes == {}

    def test_get_nodes_for_file(self, indexed_adapter: KuzuAdapter) -> None:
        nodes = indexed_adapter.get_nodes_for_file("/project/service.py")
        ids = {n.id for n in nodes}
        assert "file-svc" in ids
        assert "func-process" in ids
        assert "class-svc" in ids
        # Nodes from other.py should NOT be included
        assert "func-other" not in ids

    def test_get_nodes_for_file_nonexistent(self, indexed_adapter: KuzuAdapter) -> None:
        nodes = indexed_adapter.get_nodes_for_file("/nonexistent.py")
        assert nodes == []

    def test_get_chunk_ids_for_file(self, indexed_adapter: KuzuAdapter) -> None:
        chunk_ids = indexed_adapter.get_chunk_ids_for_file("/project/service.py")
        assert "chunk-001" in chunk_ids
        assert "chunk-002" in chunk_ids

    def test_get_chunk_ids_for_file_no_chunks(self, indexed_adapter: KuzuAdapter) -> None:
        chunk_ids = indexed_adapter.get_chunk_ids_for_file("/project/other.py")
        assert chunk_ids == []

    def test_delete_nodes_for_file(self, indexed_adapter: KuzuAdapter) -> None:
        indexed_adapter.delete_nodes_for_file("/project/service.py")
        # All service.py nodes should be gone
        assert indexed_adapter.get_node("file-svc") is None
        assert indexed_adapter.get_node("func-process") is None
        assert indexed_adapter.get_node("class-svc") is None
        assert indexed_adapter.get_node("chunk-001") is None
        assert indexed_adapter.get_node("chunk-002") is None
        # Other file nodes should remain
        assert indexed_adapter.get_node("file-other") is not None
        assert indexed_adapter.get_node("func-other") is not None

    def test_delete_nodes_for_file_nonexistent(self, indexed_adapter: KuzuAdapter) -> None:
        """Deleting nodes for a non-existent file should not raise."""
        indexed_adapter.delete_nodes_for_file("/nonexistent.py")  # Should not raise


class TestKuzuTraversal:
    """Tests for graph traversal and raw query operations."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> Generator[KuzuAdapter, None, None]:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    @pytest.fixture
    def graph_with_chain(self, adapter: KuzuAdapter) -> KuzuAdapter:
        """Create a call chain: f1 -> calls -> f2 -> calls -> f3 -> calls -> f4."""
        for i in range(1, 5):
            adapter.add_node(
                NodeData(
                    id=f"func-{i}",
                    node_type="Function",
                    properties={"name": f"func_{i}", "file": "chain.py"},
                )
            )
        adapter.add_edge(EdgeData(source_id="func-1", target_id="func-2", edge_type="CALLS"))
        adapter.add_edge(EdgeData(source_id="func-2", target_id="func-3", edge_type="CALLS"))
        adapter.add_edge(EdgeData(source_id="func-3", target_id="func-4", edge_type="CALLS"))
        return adapter

    def test_traverse_depth_1(self, graph_with_chain: KuzuAdapter) -> None:
        result = graph_with_chain.traverse("func-1", max_depth=1)
        ids = {n.id for n in result.nodes}
        assert "func-1" in ids
        assert "func-2" in ids
        assert "func-3" not in ids  # depth 2

    def test_traverse_depth_2(self, graph_with_chain: KuzuAdapter) -> None:
        result = graph_with_chain.traverse("func-1", max_depth=2)
        ids = {n.id for n in result.nodes}
        assert "func-1" in ids
        assert "func-2" in ids
        assert "func-3" in ids
        assert "func-4" not in ids  # depth 3

    def test_traverse_depth_3(self, graph_with_chain: KuzuAdapter) -> None:
        result = graph_with_chain.traverse("func-1", max_depth=3)
        ids = {n.id for n in result.nodes}
        assert "func-4" in ids  # depth 3 reached

    def test_traverse_nonexistent_start(self, adapter: KuzuAdapter) -> None:
        result = adapter.traverse("ghost")
        assert result.nodes == []

    def test_traverse_returns_traversal_result(self, graph_with_chain: KuzuAdapter) -> None:
        """Traverse returns a proper TraversalResult instance."""
        result = graph_with_chain.traverse("func-1", max_depth=1)
        assert isinstance(result, TraversalResult)

    def test_raw_query(self, graph_with_chain: KuzuAdapter) -> None:
        rows = graph_with_chain.query(
            "MATCH (n:Function) WHERE n.name = $name RETURN n.id, n.name",
            parameters={"name": "func_2"},
        )
        assert len(rows) == 1
        assert rows[0]["n.id"] == "func-2"
        assert rows[0]["n.name"] == "func_2"

    def test_raw_query_no_results(self, adapter: KuzuAdapter) -> None:
        rows = adapter.query("MATCH (n:Function) RETURN n.id")
        assert rows == []

    def test_clear_removes_everything(self, graph_with_chain: KuzuAdapter) -> None:
        graph_with_chain.clear()
        assert graph_with_chain.get_node("func-1") is None
        assert graph_with_chain.get_node("func-2") is None
        rows = graph_with_chain.query("MATCH (n:Function) RETURN n.id")
        assert rows == []
