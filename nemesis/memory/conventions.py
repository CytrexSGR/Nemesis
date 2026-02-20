"""ConventionManager -- CRUD for coding convention nodes in the knowledge graph."""

from __future__ import annotations

from datetime import datetime

from nemesis.graph.adapter import EdgeData, GraphAdapter, NodeData
from nemesis.memory.models import ConventionModel


class ConventionManager:
    """Manage Convention nodes in the knowledge graph."""

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    def add_convention(
        self,
        pattern: str,
        example: str = "",
        scope: str = "project",
    ) -> ConventionModel:
        """Create a new Convention node and return its model."""
        model = ConventionModel(pattern=pattern, example=example, scope=scope)
        node = NodeData(
            id=model.id,
            node_type="Convention",
            properties={
                "pattern": model.pattern,
                "example": model.example,
                "scope": model.scope,
                "created_at": model.created_at.isoformat(),
            },
        )
        self._graph.add_node(node)
        return model

    def get_conventions(self) -> list[ConventionModel]:
        """Return all Convention nodes as models."""
        rows = self._graph.query("MATCH (c:Convention) RETURN c")
        return [self._row_to_model(row) for row in rows]

    def link_governs(self, convention_id: str, target_id: str) -> None:
        """Create a GOVERNS edge from a Convention to a target node."""
        edge = EdgeData(
            source_id=convention_id,
            target_id=target_id,
            edge_type="GOVERNS",
        )
        self._graph.add_edge(edge)

    def delete_convention(self, convention_id: str) -> bool:
        """Delete a convention by its ID.

        Returns:
            True if the convention existed and was deleted, False otherwise.
        """
        node = self._graph.get_node(convention_id)
        if node is None:
            return False
        self._graph.delete_node(convention_id)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_model(row: dict) -> ConventionModel:
        """Convert a query result row (dict) to a ConventionModel."""
        data = row.get("c", row)
        created_at = data.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            from datetime import UTC

            created_at = datetime.now(UTC)

        return ConventionModel(
            id=str(data.get("id", "")),
            pattern=str(data.get("pattern", "")),
            example=str(data.get("example", "")),
            scope=str(data.get("scope", "project")),
            created_at=created_at,
        )
