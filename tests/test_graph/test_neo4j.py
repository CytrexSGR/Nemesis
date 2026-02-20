"""Tests for Neo4j graph adapter.

These tests use a mock Neo4j driver since a running Neo4j instance
is not required for development. Integration tests with a real
Neo4j server are marked with @pytest.mark.integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nemesis.graph.adapter import EdgeData, GraphAdapter, NodeData, TraversalResult
from nemesis.graph.neo4j import Neo4jAdapter


class TestNeo4jInit:
    """Tests for Neo4jAdapter initialization."""

    def test_is_graph_adapter(self) -> None:
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_driver.GraphDatabase.driver.return_value = MagicMock()
            adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
            assert isinstance(adapter, GraphAdapter)
            adapter.close()

    def test_create_schema_idempotent(self) -> None:
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_session = MagicMock()
            mock_driver_instance = MagicMock()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_driver_instance.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
            adapter.create_schema()
            adapter.create_schema()  # Should not raise
            assert mock_session.run.called
            adapter.close()

    def test_import_error_when_driver_missing(self) -> None:
        with (
            patch("nemesis.graph.neo4j.neo4j_driver", None),
            pytest.raises(ImportError, match="neo4j driver is required"),
        ):
            Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")

    def test_close_sets_driver_to_none(self) -> None:
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_driver_instance = MagicMock()
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
            adapter.close()
            assert adapter._driver is None
            mock_driver_instance.close.assert_called_once()


class TestNeo4jNodeCRUD:
    """Tests for Neo4j node CRUD with mocked driver."""

    @pytest.fixture
    def mock_neo4j(self):
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_session = MagicMock()
            mock_tx = MagicMock()
            mock_driver_instance = MagicMock()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_driver_instance.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.execute_write.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_session.execute_read.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            yield {
                "driver": mock_driver,
                "driver_instance": mock_driver_instance,
                "session": mock_session,
                "tx": mock_tx,
            }

    def test_add_node_executes_merge(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.add_node(NodeData(id="f1", node_type="Function", properties={"name": "hello"}))
        # Verify that tx.run was called with a MERGE query
        assert mock_neo4j["tx"].run.called
        call_args = mock_neo4j["tx"].run.call_args
        assert "MERGE" in call_args[0][0]
        adapter.close()

    def test_add_node_merge_includes_node_type(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.add_node(NodeData(id="c1", node_type="Class", properties={"name": "Foo"}))
        call_args = mock_neo4j["tx"].run.call_args
        query = call_args[0][0]
        assert "Class" in query
        assert "MERGE" in query
        adapter.close()

    def test_add_node_sets_properties(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.add_node(
            NodeData(
                id="f1",
                node_type="Function",
                properties={"name": "hello", "file": "main.py"},
            )
        )
        call_args = mock_neo4j["tx"].run.call_args
        query = call_args[0][0]
        assert "SET" in query
        assert "n.name" in query
        assert "n.file" in query
        adapter.close()

    def test_get_node_executes_match(self, mock_neo4j) -> None:
        mock_record = MagicMock()
        mock_record.data.return_value = {"id": "f1", "name": "hello"}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.get_node("f1")
        assert mock_neo4j["tx"].run.called
        adapter.close()

    def test_get_node_returns_none_when_not_found(self, mock_neo4j) -> None:
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        node = adapter.get_node("nonexistent")
        assert node is None
        adapter.close()

    def test_get_node_returns_node_data(self, mock_neo4j) -> None:
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "labels": ["Function"],
            "props": {"id": "f1", "name": "hello"},
        }
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        node = adapter.get_node("f1")
        assert node is not None
        assert node.id == "f1"
        assert node.node_type == "Function"
        assert node.properties["name"] == "hello"
        adapter.close()

    def test_delete_node_executes_detach_delete(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.delete_node("f1")
        assert mock_neo4j["tx"].run.called
        call_args = mock_neo4j["tx"].run.call_args
        assert "DETACH DELETE" in call_args[0][0]
        adapter.close()


class TestNeo4jEdgeCRUD:
    """Tests for Neo4j edge operations with mocked driver."""

    @pytest.fixture
    def mock_neo4j(self):
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_session = MagicMock()
            mock_tx = MagicMock()
            mock_driver_instance = MagicMock()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_driver_instance.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.execute_write.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_session.execute_read.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            yield {
                "driver": mock_driver,
                "driver_instance": mock_driver_instance,
                "session": mock_session,
                "tx": mock_tx,
            }

    def test_add_edge_executes_merge(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.add_edge(EdgeData(source_id="f1", target_id="f2", edge_type="CALLS"))
        assert mock_neo4j["tx"].run.called
        call_args = mock_neo4j["tx"].run.call_args
        query = call_args[0][0]
        assert "MATCH" in query
        assert "CALLS" in query
        adapter.close()

    def test_add_edge_with_properties(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.add_edge(
            EdgeData(
                source_id="f1",
                target_id="f2",
                edge_type="CALLS",
                properties={"weight": 1.0},
            )
        )
        call_args = mock_neo4j["tx"].run.call_args
        query = call_args[0][0]
        assert "weight" in query
        adapter.close()

    def test_delete_edges_for_file(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.delete_edges_for_file("/src/main.py")
        assert mock_neo4j["tx"].run.called
        adapter.close()


class TestNeo4jNeighborsAndTraversal:
    """Tests for Neo4j neighbor and traversal operations."""

    @pytest.fixture
    def mock_neo4j(self):
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_session = MagicMock()
            mock_tx = MagicMock()
            mock_driver_instance = MagicMock()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_driver_instance.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.execute_write.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_session.execute_read.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            yield {
                "driver": mock_driver,
                "driver_instance": mock_driver_instance,
                "session": mock_session,
                "tx": mock_tx,
            }

    def test_get_neighbors_outgoing(self, mock_neo4j) -> None:
        mock_record1 = MagicMock()
        mock_record1.data.return_value = {
            "labels": ["Function"],
            "props": {"id": "f2", "name": "target"},
        }
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([mock_record1]))
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        neighbors = adapter.get_neighbors("f1", direction="outgoing")
        assert len(neighbors) == 1
        assert neighbors[0].id == "f2"
        adapter.close()

    def test_get_neighbors_incoming(self, mock_neo4j) -> None:
        mock_record1 = MagicMock()
        mock_record1.data.return_value = {
            "labels": ["File"],
            "props": {"id": "file1", "path": "main.py"},
        }
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([mock_record1]))
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        neighbors = adapter.get_neighbors("f1", direction="incoming")
        assert len(neighbors) == 1
        assert neighbors[0].id == "file1"
        adapter.close()

    def test_get_neighbors_with_edge_type_filter(self, mock_neo4j) -> None:
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.get_neighbors("f1", edge_type="CALLS", direction="outgoing")
        call_args = mock_neo4j["tx"].run.call_args
        query = call_args[0][0]
        assert "CALLS" in query
        adapter.close()

    def test_get_neighbors_both_direction(self, mock_neo4j) -> None:
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.get_neighbors("f1", direction="both")
        call_args = mock_neo4j["tx"].run.call_args
        query = call_args[0][0]
        assert "DISTINCT" in query
        adapter.close()

    def test_traverse_returns_traversal_result(self, mock_neo4j) -> None:
        # Mock get_node for the start node
        mock_record_start = MagicMock()
        mock_record_start.data.return_value = {
            "labels": ["Function"],
            "props": {"id": "f1", "name": "start"},
        }
        mock_result_start = MagicMock()
        mock_result_start.single.return_value = mock_record_start

        # Mock traversal query returns empty list
        mock_result_traverse = MagicMock()
        mock_result_traverse.__iter__ = MagicMock(return_value=iter([]))

        mock_neo4j["tx"].run.side_effect = [
            mock_result_start,
            mock_result_traverse,
        ]

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        result = adapter.traverse("f1", max_depth=2)
        assert isinstance(result, TraversalResult)
        assert len(result.nodes) >= 1  # At least the start node
        adapter.close()

    def test_traverse_with_edge_types(self, mock_neo4j) -> None:
        mock_record_start = MagicMock()
        mock_record_start.data.return_value = {
            "labels": ["Function"],
            "props": {"id": "f1", "name": "start"},
        }
        mock_result_start = MagicMock()
        mock_result_start.single.return_value = mock_record_start

        mock_result_traverse = MagicMock()
        mock_result_traverse.__iter__ = MagicMock(return_value=iter([]))

        mock_neo4j["tx"].run.side_effect = [
            mock_result_start,
            mock_result_traverse,
        ]

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        result = adapter.traverse("f1", edge_types=["CALLS", "CONTAINS"], max_depth=2)
        assert isinstance(result, TraversalResult)
        adapter.close()


class TestNeo4jFileOperations:
    """Tests for file-related operations."""

    @pytest.fixture
    def mock_neo4j(self):
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_session = MagicMock()
            mock_tx = MagicMock()
            mock_driver_instance = MagicMock()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_driver_instance.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.execute_write.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_session.execute_read.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            yield {
                "driver": mock_driver,
                "driver_instance": mock_driver_instance,
                "session": mock_session,
                "tx": mock_tx,
            }

    def test_get_file_hashes(self, mock_neo4j) -> None:
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter([{"path": "/src/main.py", "hash": "abc123"}])
        )
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.get_file_hashes()
        assert mock_neo4j["tx"].run.called
        adapter.close()

    def test_get_nodes_for_file(self, mock_neo4j) -> None:
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter([{"labels": ["Function"], "props": {"id": "f1", "name": "hello"}}])
        )
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.get_nodes_for_file("/src/main.py")
        assert mock_neo4j["tx"].run.called
        adapter.close()

    def test_get_chunk_ids_for_file(self, mock_neo4j) -> None:
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([{"id": "chunk-001"}]))
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.get_chunk_ids_for_file("/src/main.py")
        assert mock_neo4j["tx"].run.called
        adapter.close()

    def test_delete_nodes_for_file(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.delete_nodes_for_file("/src/main.py")
        # Should have called tx.run at least twice (chunk delete + node delete)
        assert mock_neo4j["tx"].run.call_count >= 2
        adapter.close()

    def test_clear(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.clear()
        call_args = mock_neo4j["tx"].run.call_args
        query = call_args[0][0]
        assert "DETACH DELETE" in query
        adapter.close()

    def test_raw_query(self, mock_neo4j) -> None:
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([{"n.id": "f1", "n.name": "hello"}]))
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.query(
            "MATCH (n:Function) RETURN n.id, n.name",
            parameters={"name": "hello"},
        )
        assert mock_neo4j["tx"].run.called
        adapter.close()
