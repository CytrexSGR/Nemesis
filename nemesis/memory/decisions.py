"""DecisionsManager -- CRUD for architectural decisions and alternatives."""

from __future__ import annotations

from datetime import datetime

from nemesis.graph.adapter import EdgeData, GraphAdapter, NodeData
from nemesis.memory.models import AlternativeModel, DecisionModel


class DecisionsManager:
    """Manage Decision and Alternative nodes in the knowledge graph."""

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def add_decision(
        self,
        title: str,
        reasoning: str = "",
        status: str = "proposed",
    ) -> DecisionModel:
        """Create a new Decision node and return its model."""
        model = DecisionModel(title=title, reasoning=reasoning, status=status)
        node = NodeData(
            id=model.id,
            node_type="Decision",
            properties={
                "title": model.title,
                "reasoning": model.reasoning,
                "created_at": model.created_at.isoformat(),
                "status": model.status,
            },
        )
        self._graph.add_node(node)
        return model

    def get_decisions(self) -> list[DecisionModel]:
        """Return all Decision nodes as models."""
        rows = self._graph.query("MATCH (d:Decision) RETURN d")
        return [
            DecisionModel(
                id=row["d"]["id"],
                title=row["d"]["title"],
                reasoning=row["d"].get("reasoning", ""),
                status=row["d"].get("status", "proposed"),
                created_at=datetime.fromisoformat(row["d"]["created_at"]),
            )
            for row in rows
        ]

    def update_decision_status(
        self,
        decision_id: str,
        status: str,
    ) -> DecisionModel | None:
        """Update the status of an existing Decision. Returns None if not found."""
        existing = self._graph.get_node(decision_id)
        if existing is None:
            return None

        props = dict(existing.properties)
        props["status"] = status
        updated_node = NodeData(
            id=existing.id,
            node_type=existing.node_type,
            properties=props,
        )
        self._graph.add_node(updated_node)
        return DecisionModel(
            id=existing.id,
            title=props["title"],
            reasoning=props.get("reasoning", ""),
            status=status,
            created_at=datetime.fromisoformat(props["created_at"]),
        )

    def delete_decision(self, decision_id: str) -> bool:
        """Delete a decision and all its linked alternatives."""
        alternatives = self._graph.get_neighbors(
            decision_id,
            edge_type="REJECTED",
            direction="outgoing",
        )
        for alt in alternatives:
            self._graph.delete_node(alt.id)
        self._graph.delete_node(decision_id)
        return True

    # ------------------------------------------------------------------
    # Alternatives
    # ------------------------------------------------------------------

    def add_alternative(
        self,
        decision_id: str,
        title: str,
        reason_rejected: str = "",
    ) -> AlternativeModel:
        """Create an Alternative node and link it via REJECTED edge."""
        model = AlternativeModel(title=title, reason_rejected=reason_rejected)
        node = NodeData(
            id=model.id,
            node_type="Alternative",
            properties={
                "title": model.title,
                "reason_rejected": model.reason_rejected,
            },
        )
        self._graph.add_node(node)
        edge = EdgeData(
            source_id=decision_id,
            target_id=model.id,
            edge_type="REJECTED",
        )
        self._graph.add_edge(edge)
        return model

    def add_chose_link(self, decision_id: str, target_id: str) -> None:
        """Create a CHOSE edge from a Decision to a target node."""
        edge = EdgeData(
            source_id=decision_id,
            target_id=target_id,
            edge_type="CHOSE",
        )
        self._graph.add_edge(edge)

    # ------------------------------------------------------------------
    # Combined queries
    # ------------------------------------------------------------------

    def get_decision_with_alternatives(
        self,
        decision_id: str,
    ) -> tuple[DecisionModel | None, list[AlternativeModel]]:
        """Return a decision and all its rejected alternatives."""
        existing = self._graph.get_node(decision_id)
        if existing is None:
            return None, []

        props = existing.properties
        decision = DecisionModel(
            id=existing.id,
            title=props["title"],
            reasoning=props.get("reasoning", ""),
            status=props.get("status", "proposed"),
            created_at=datetime.fromisoformat(props["created_at"]),
        )

        neighbors = self._graph.get_neighbors(
            decision_id,
            edge_type="REJECTED",
            direction="outgoing",
        )
        alternatives = [
            AlternativeModel(
                id=n.id,
                title=n.properties["title"],
                reason_rejected=n.properties.get("reason_rejected", ""),
            )
            for n in neighbors
        ]
        return decision, alternatives
