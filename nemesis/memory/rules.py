"""RulesManager — CRUD operations for Rule nodes in the knowledge graph."""

from __future__ import annotations

from datetime import UTC, datetime

from nemesis.graph.adapter import EdgeData, GraphAdapter, NodeData
from nemesis.memory.models import RuleModel


class RulesManager:
    """Manages Rule nodes in the graph database.

    Provides create, read, update, delete (CRUD) operations for rules
    and supports scoping and linking rules to targets.
    """

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    def add_rule(
        self,
        content: str,
        scope: str = "project",
        source: str = "manual",
    ) -> RuleModel:
        """Add a new rule to the graph.

        Args:
            content: The rule text.
            scope: Scope of the rule (e.g. "project", "file").
            source: Origin of the rule (e.g. "manual", "auto").

        Returns:
            The created RuleModel.
        """
        model = RuleModel(content=content, scope=scope, source=source)
        node = NodeData(
            id=model.id,
            node_type="Rule",
            properties={
                "content": model.content,
                "scope": model.scope,
                "created_at": model.created_at.isoformat(),
                "source": model.source,
            },
        )
        self._graph.add_node(node)
        return model

    def get_rules(self) -> list[RuleModel]:
        """Retrieve all rules from the graph.

        Returns:
            List of all RuleModel instances.
        """
        rows = self._graph.query("MATCH (r:Rule) RETURN r")
        return [self._row_to_model(row) for row in rows]

    def get_rule_by_id(self, rule_id: str) -> RuleModel | None:
        """Retrieve a single rule by its ID.

        Args:
            rule_id: The unique rule identifier.

        Returns:
            RuleModel if found, None otherwise.
        """
        node = self._graph.get_node(rule_id)
        if node is None:
            return None
        return self._node_to_model(node)

    def update_rule(self, rule_id: str, **kwargs: object) -> RuleModel | None:
        """Update an existing rule.

        Args:
            rule_id: The ID of the rule to update.
            **kwargs: Fields to update (content, scope, source).

        Returns:
            Updated RuleModel if found, None otherwise.
        """
        node = self._graph.get_node(rule_id)
        if node is None:
            return None

        props = dict(node.properties)
        props.update(kwargs)

        updated_node = NodeData(
            id=rule_id,
            node_type="Rule",
            properties=props,
        )
        self._graph.add_node(updated_node)
        return self._node_to_model(updated_node)

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by its ID.

        Args:
            rule_id: The ID of the rule to delete.

        Returns:
            True if the rule existed and was deleted, False otherwise.
        """
        node = self._graph.get_node(rule_id)
        if node is None:
            return False
        self._graph.delete_node(rule_id)
        return True

    def get_rules_for_scope(self, scope: str) -> list[RuleModel]:
        """Retrieve all rules matching a specific scope.

        Args:
            scope: The scope to filter by.

        Returns:
            List of matching RuleModel instances.
        """
        rows = self._graph.query(
            "MATCH (r:Rule) WHERE r.scope = $scope RETURN r",
            parameters={"scope": scope},
        )
        return [self._row_to_model(row) for row in rows]

    def link_rule_to_target(self, rule_id: str, target_id: str) -> None:
        """Create an APPLIES_TO edge from a rule to a target node.

        Args:
            rule_id: The source rule ID.
            target_id: The target node ID.
        """
        edge = EdgeData(
            source_id=rule_id,
            target_id=target_id,
            edge_type="APPLIES_TO",
        )
        self._graph.add_edge(edge)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_to_model(node: NodeData) -> RuleModel:
        """Convert a NodeData instance to a RuleModel."""
        props = node.properties
        created_at = props.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now(UTC)

        return RuleModel(
            id=node.id,
            content=str(props.get("content", "")),
            scope=str(props.get("scope", "project")),
            source=str(props.get("source", "manual")),
            created_at=created_at,
        )

    @staticmethod
    def _row_to_model(row: dict) -> RuleModel:
        """Convert a query result row (dict) to a RuleModel."""
        data = row.get("r", row)
        created_at = data.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now(UTC)

        return RuleModel(
            id=str(data.get("id", "")),
            content=str(data.get("content", "")),
            scope=str(data.get("scope", "project")),
            source=str(data.get("source", "manual")),
            created_at=created_at,
        )
