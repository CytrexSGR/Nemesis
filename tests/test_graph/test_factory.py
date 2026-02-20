"""Tests for the graph module factory function."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from nemesis.graph import create_graph_adapter
from nemesis.graph.adapter import GraphAdapter


class TestCreateGraphAdapter:
    def test_create_kuzu_adapter(self, tmp_path: Path) -> None:
        adapter = create_graph_adapter(backend="kuzu", db_path=str(tmp_path / "test_graph"))
        assert isinstance(adapter, GraphAdapter)
        adapter.close()

    def test_create_kuzu_is_default(self, tmp_path: Path) -> None:
        adapter = create_graph_adapter(db_path=str(tmp_path / "test_graph"))
        from nemesis.graph.kuzu import KuzuAdapter

        assert isinstance(adapter, KuzuAdapter)
        adapter.close()

    def test_create_neo4j_adapter(self) -> None:
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_driver.GraphDatabase.driver.return_value = MagicMock()
            adapter = create_graph_adapter(
                backend="neo4j",
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test",
            )
            assert isinstance(adapter, GraphAdapter)
            adapter.close()

    def test_create_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown graph backend"):
            create_graph_adapter(backend="invalid")

    def test_kuzu_adapter_creates_schema(self, tmp_path: Path) -> None:
        adapter = create_graph_adapter(
            backend="kuzu", db_path=str(tmp_path / "test_graph"), create_schema=True
        )
        from nemesis.graph.adapter import NodeData

        adapter.add_node(NodeData(id="test", node_type="File", properties={"path": "test.py"}))
        assert adapter.get_node("test") is not None
        adapter.close()
