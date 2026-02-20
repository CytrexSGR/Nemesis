"""Tests for nemesis.memory.conventions.ConventionManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nemesis.graph.adapter import EdgeData, NodeData
from nemesis.memory.conventions import ConventionManager
from nemesis.memory.models import ConventionModel


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return ConventionManager(mock_graph)


# ──────────────────────────────────────────────────────────────────────
# test_add_convention
# ──────────────────────────────────────────────────────────────────────


class TestAddConvention:
    def test_add_convention(self, manager, mock_graph) -> None:
        """Creates a Convention node with correct properties."""
        result = manager.add_convention(
            pattern="snake_case for functions",
            example="def my_function():",
            scope="project",
        )

        assert isinstance(result, ConventionModel)
        assert result.pattern == "snake_case for functions"
        assert result.example == "def my_function():"
        assert result.scope == "project"
        assert result.id is not None

        mock_graph.add_node.assert_called_once()
        node: NodeData = mock_graph.add_node.call_args[0][0]
        assert node.node_type == "Convention"
        assert node.properties["pattern"] == "snake_case for functions"
        assert node.properties["example"] == "def my_function():"
        assert node.properties["scope"] == "project"
        assert "created_at" in node.properties


# ──────────────────────────────────────────────────────────────────────
# test_get_conventions
# ──────────────────────────────────────────────────────────────────────


class TestGetConventions:
    def test_get_conventions(self, manager, mock_graph) -> None:
        """Mock query returns rows, converts to ConventionModels."""
        mock_graph.query.return_value = [
            {
                "c": {
                    "id": "conv-1",
                    "pattern": "PascalCase for classes",
                    "example": "class MyClass:",
                    "scope": "project",
                    "created_at": "2025-01-01T00:00:00+00:00",
                }
            },
            {
                "c": {
                    "id": "conv-2",
                    "pattern": "ALL_CAPS for constants",
                    "example": "MAX_RETRIES = 3",
                    "scope": "file",
                    "created_at": "2025-02-01T00:00:00+00:00",
                }
            },
        ]

        conventions = manager.get_conventions()

        assert len(conventions) == 2
        assert all(isinstance(c, ConventionModel) for c in conventions)
        assert conventions[0].pattern == "PascalCase for classes"
        assert conventions[1].pattern == "ALL_CAPS for constants"
        assert conventions[1].scope == "file"


# ──────────────────────────────────────────────────────────────────────
# test_link_convention_governs
# ──────────────────────────────────────────────────────────────────────


class TestLinkConventionGoverns:
    def test_link_governs(self, manager, mock_graph) -> None:
        """Creates a GOVERNS edge from convention to target."""
        manager.link_governs("conv-1", "target-1")

        mock_graph.add_edge.assert_called_once()
        edge: EdgeData = mock_graph.add_edge.call_args[0][0]
        assert edge.source_id == "conv-1"
        assert edge.target_id == "target-1"
        assert edge.edge_type == "GOVERNS"


# ──────────────────────────────────────────────────────────────────────
# test_delete_convention
# ──────────────────────────────────────────────────────────────────────


class TestDeleteConvention:
    def test_delete_existing_convention(self, manager, mock_graph) -> None:
        """Deletes node and returns True for existing convention."""
        mock_graph.get_node.return_value = NodeData(
            id="conv-1",
            node_type="Convention",
            properties={"pattern": "To delete", "scope": "project"},
        )

        assert manager.delete_convention("conv-1") is True
        mock_graph.delete_node.assert_called_once_with("conv-1")

    def test_delete_nonexistent_returns_false(self, manager, mock_graph) -> None:
        """Returns False for nonexistent convention."""
        mock_graph.get_node.return_value = None
        assert manager.delete_convention("nonexistent") is False
