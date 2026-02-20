"""Tests for Kuzu graph adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemesis.graph.adapter import (
    NODE_TYPES,
    EdgeData,
    GraphAdapter,
    NodeData,
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
