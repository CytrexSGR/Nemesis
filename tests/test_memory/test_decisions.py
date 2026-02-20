"""Tests for nemesis.memory.decisions.DecisionsManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nemesis.graph.adapter import NodeData
from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.models import AlternativeModel, DecisionModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return DecisionsManager(mock_graph)


# ---------------------------------------------------------------------------
# TestAddDecision
# ---------------------------------------------------------------------------


class TestAddDecision:
    def test_add_decision(self, manager, mock_graph):
        result = manager.add_decision(
            title="Use Kuzu",
            reasoning="Embedded graph DB",
            status="accepted",
        )
        assert isinstance(result, DecisionModel)
        assert result.title == "Use Kuzu"
        assert result.reasoning == "Embedded graph DB"
        assert result.status == "accepted"
        assert result.id is not None

        mock_graph.add_node.assert_called_once()
        node_arg = mock_graph.add_node.call_args[0][0]
        assert node_arg.node_type == "Decision"
        assert node_arg.properties["title"] == "Use Kuzu"

    def test_add_decision_default_status(self, manager):
        result = manager.add_decision(title="Some decision")
        assert result.status == "proposed"


# ---------------------------------------------------------------------------
# TestGetDecisions
# ---------------------------------------------------------------------------


class TestGetDecisions:
    def test_get_decisions(self, manager, mock_graph):
        mock_graph.query.return_value = [
            {
                "d": {
                    "id": "dec-1",
                    "title": "Use FastAPI",
                    "reasoning": "Modern async",
                    "status": "accepted",
                    "created_at": "2025-01-01T00:00:00+00:00",
                }
            },
            {
                "d": {
                    "id": "dec-2",
                    "title": "Use Click",
                    "reasoning": "CLI standard",
                    "status": "proposed",
                    "created_at": "2025-02-01T00:00:00+00:00",
                }
            },
        ]
        results = manager.get_decisions()
        assert len(results) == 2
        assert all(isinstance(r, DecisionModel) for r in results)
        assert results[0].title == "Use FastAPI"
        assert results[1].title == "Use Click"


# ---------------------------------------------------------------------------
# TestAddAlternative
# ---------------------------------------------------------------------------


class TestAddAlternative:
    def test_add_alternative(self, manager, mock_graph):
        result = manager.add_alternative(
            decision_id="dec-1",
            title="MySQL",
            reason_rejected="No graph support",
        )
        assert isinstance(result, AlternativeModel)
        assert result.title == "MySQL"
        assert result.reason_rejected == "No graph support"

        mock_graph.add_node.assert_called_once()
        node_arg = mock_graph.add_node.call_args[0][0]
        assert node_arg.node_type == "Alternative"
        assert node_arg.properties["title"] == "MySQL"

        mock_graph.add_edge.assert_called_once()
        edge_arg = mock_graph.add_edge.call_args[0][0]
        assert edge_arg.source_id == "dec-1"
        assert edge_arg.target_id == result.id
        assert edge_arg.edge_type == "REJECTED"

    def test_add_alternative_chose_edge(self, manager, mock_graph):
        manager.add_chose_link(decision_id="dec-1", target_id="target-1")
        mock_graph.add_edge.assert_called_once()
        edge_arg = mock_graph.add_edge.call_args[0][0]
        assert edge_arg.source_id == "dec-1"
        assert edge_arg.target_id == "target-1"
        assert edge_arg.edge_type == "CHOSE"


# ---------------------------------------------------------------------------
# TestUpdateDecisionStatus
# ---------------------------------------------------------------------------


class TestUpdateDecisionStatus:
    def test_update_status(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="dec-1",
            node_type="Decision",
            properties={
                "title": "Use Kuzu",
                "reasoning": "Embedded",
                "created_at": "2025-01-01T00:00:00+00:00",
                "status": "proposed",
            },
        )
        result = manager.update_decision_status("dec-1", "accepted")
        assert result is not None
        assert result.status == "accepted"
        assert result.title == "Use Kuzu"

        mock_graph.add_node.assert_called_once()
        updated_node = mock_graph.add_node.call_args[0][0]
        assert updated_node.properties["status"] == "accepted"

    def test_update_nonexistent_returns_none(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        result = manager.update_decision_status("no-such-id", "accepted")
        assert result is None


# ---------------------------------------------------------------------------
# TestGetDecisionWithAlternatives
# ---------------------------------------------------------------------------


class TestGetDecisionWithAlternatives:
    def test_get_with_alternatives(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="dec-1",
            node_type="Decision",
            properties={
                "title": "Use Kuzu",
                "reasoning": "Embedded",
                "created_at": "2025-01-01T00:00:00+00:00",
                "status": "accepted",
            },
        )
        mock_graph.get_neighbors.return_value = [
            NodeData(
                id="alt-1",
                node_type="Alternative",
                properties={"title": "MySQL", "reason_rejected": "No graph"},
            ),
        ]
        decision, alternatives = manager.get_decision_with_alternatives("dec-1")
        assert decision is not None
        assert isinstance(decision, DecisionModel)
        assert decision.title == "Use Kuzu"
        assert len(alternatives) == 1
        assert isinstance(alternatives[0], AlternativeModel)
        assert alternatives[0].title == "MySQL"


# ---------------------------------------------------------------------------
# TestDeleteDecision
# ---------------------------------------------------------------------------


class TestDeleteDecision:
    def test_delete_with_alternatives(self, manager, mock_graph):
        mock_graph.get_neighbors.return_value = [
            NodeData(id="alt-1", node_type="Alternative", properties={"title": "X"}),
        ]
        result = manager.delete_decision("dec-1")
        assert result is True
        assert mock_graph.delete_node.call_count == 2
        mock_graph.delete_node.assert_any_call("alt-1")
        mock_graph.delete_node.assert_any_call("dec-1")
