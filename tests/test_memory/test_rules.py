"""Tests for nemesis.memory.rules.RulesManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nemesis.graph.adapter import EdgeData, NodeData
from nemesis.memory.models import RuleModel
from nemesis.memory.rules import RulesManager


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return RulesManager(mock_graph)


# ──────────────────────────────────────────────────────────────────────
# TestAddRule
# ──────────────────────────────────────────────────────────────────────


class TestAddRule:
    def test_add_rule_calls_add_node(self, manager, mock_graph) -> None:
        """add_rule creates a Rule node with correct type and content."""
        manager.add_rule(content="Always use type hints", scope="project", source="manual")

        mock_graph.add_node.assert_called_once()
        node: NodeData = mock_graph.add_node.call_args[0][0]
        assert node.node_type == "Rule"
        assert node.properties["content"] == "Always use type hints"
        assert node.properties["scope"] == "project"
        assert node.properties["source"] == "manual"
        assert "created_at" in node.properties

    def test_add_rule_returns_model(self, manager) -> None:
        """add_rule returns a RuleModel with correct fields."""
        result = manager.add_rule(content="Use black", scope="file", source="auto")

        assert isinstance(result, RuleModel)
        assert result.content == "Use black"
        assert result.scope == "file"
        assert result.source == "auto"
        assert result.id is not None


# ──────────────────────────────────────────────────────────────────────
# TestGetRules
# ──────────────────────────────────────────────────────────────────────


class TestGetRules:
    def test_get_rules_queries_graph(self, manager, mock_graph) -> None:
        """get_rules queries the graph and returns RuleModels."""
        mock_graph.query.return_value = [
            {
                "r": {
                    "id": "r1",
                    "content": "Rule one",
                    "scope": "project",
                    "source": "manual",
                    "created_at": "2025-01-01T00:00:00+00:00",
                }
            },
            {
                "r": {
                    "id": "r2",
                    "content": "Rule two",
                    "scope": "file",
                    "source": "auto",
                    "created_at": "2025-01-02T00:00:00+00:00",
                }
            },
        ]

        rules = manager.get_rules()

        assert len(rules) == 2
        assert all(isinstance(r, RuleModel) for r in rules)
        assert rules[0].content == "Rule one"
        assert rules[1].content == "Rule two"

    def test_get_rules_empty(self, manager, mock_graph) -> None:
        """get_rules returns an empty list when no rules exist."""
        mock_graph.query.return_value = []
        assert manager.get_rules() == []


# ──────────────────────────────────────────────────────────────────────
# TestGetRuleById
# ──────────────────────────────────────────────────────────────────────


class TestGetRuleById:
    def test_get_existing_rule(self, manager, mock_graph) -> None:
        """get_rule_by_id returns a RuleModel when the node exists."""
        mock_graph.get_node.return_value = NodeData(
            id="r1",
            node_type="Rule",
            properties={
                "content": "Format with black",
                "scope": "project",
                "source": "manual",
                "created_at": "2025-01-01T00:00:00+00:00",
            },
        )

        result = manager.get_rule_by_id("r1")

        assert result is not None
        assert isinstance(result, RuleModel)
        assert result.id == "r1"
        assert result.content == "Format with black"

    def test_get_nonexistent_rule(self, manager, mock_graph) -> None:
        """get_rule_by_id returns None when the node does not exist."""
        mock_graph.get_node.return_value = None
        assert manager.get_rule_by_id("nonexistent") is None


# ──────────────────────────────────────────────────────────────────────
# TestUpdateRule
# ──────────────────────────────────────────────────────────────────────


class TestUpdateRule:
    def test_update_existing_rule(self, manager, mock_graph) -> None:
        """update_rule updates content and calls add_node."""
        mock_graph.get_node.return_value = NodeData(
            id="r1",
            node_type="Rule",
            properties={
                "content": "Old content",
                "scope": "project",
                "source": "manual",
                "created_at": "2025-01-01T00:00:00+00:00",
            },
        )

        result = manager.update_rule("r1", content="New content")

        assert result is not None
        assert result.content == "New content"
        mock_graph.add_node.assert_called_once()
        updated_node: NodeData = mock_graph.add_node.call_args[0][0]
        assert updated_node.properties["content"] == "New content"

    def test_update_nonexistent_returns_none(self, manager, mock_graph) -> None:
        """update_rule returns None when the rule does not exist."""
        mock_graph.get_node.return_value = None
        assert manager.update_rule("nonexistent", content="x") is None


# ──────────────────────────────────────────────────────────────────────
# TestDeleteRule
# ──────────────────────────────────────────────────────────────────────


class TestDeleteRule:
    def test_delete_existing_rule(self, manager, mock_graph) -> None:
        """delete_rule returns True and calls delete_node."""
        mock_graph.get_node.return_value = NodeData(
            id="r1",
            node_type="Rule",
            properties={"content": "To delete", "scope": "project"},
        )

        assert manager.delete_rule("r1") is True
        mock_graph.delete_node.assert_called_once_with("r1")

    def test_delete_nonexistent_returns_false(self, manager, mock_graph) -> None:
        """delete_rule returns False when the rule does not exist."""
        mock_graph.get_node.return_value = None
        assert manager.delete_rule("nonexistent") is False


# ──────────────────────────────────────────────────────────────────────
# TestRulesForScope
# ──────────────────────────────────────────────────────────────────────


class TestRulesForScope:
    def test_get_rules_for_scope(self, manager, mock_graph) -> None:
        """get_rules_for_scope queries with scope parameter."""
        mock_graph.query.return_value = [
            {
                "r": {
                    "id": "r1",
                    "content": "Scoped rule",
                    "scope": "file",
                    "source": "manual",
                    "created_at": "2025-01-01T00:00:00+00:00",
                }
            },
        ]

        rules = manager.get_rules_for_scope("file")

        assert len(rules) == 1
        assert rules[0].scope == "file"
        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert call_args[1]["parameters"]["scope"] == "file"


# ──────────────────────────────────────────────────────────────────────
# TestLinkRule
# ──────────────────────────────────────────────────────────────────────


class TestLinkRule:
    def test_link_rule_to_target(self, manager, mock_graph) -> None:
        """link_rule_to_target creates an APPLIES_TO edge."""
        manager.link_rule_to_target("r1", "target1")

        mock_graph.add_edge.assert_called_once()
        edge: EdgeData = mock_graph.add_edge.call_args[0][0]
        assert edge.source_id == "r1"
        assert edge.target_id == "target1"
        assert edge.edge_type == "APPLIES_TO"
