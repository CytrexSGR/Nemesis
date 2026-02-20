"""Integration tests for the Nemesis memory subsystem.

Uses a FakeGraph (in-memory dict-based) to test the full lifecycle
without requiring a real database backend.
"""

from __future__ import annotations

from nemesis.graph.adapter import EdgeData, NodeData, TraversalResult
from nemesis.memory.auto_learn import process_message
from nemesis.memory.context import SessionContext
from nemesis.memory.conventions import ConventionManager
from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.rules import RulesManager

# ===================================================================
# FakeGraph — in-memory GraphAdapter implementation for testing
# ===================================================================


class FakeGraph:
    """Minimal in-memory graph adapter for integration tests."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeData] = {}
        self._edges: list[EdgeData] = []

    def create_schema(self) -> None:
        pass

    def add_node(self, node: NodeData) -> None:
        self._nodes[node.id] = node

    def add_edge(self, edge: EdgeData) -> None:
        self._edges.append(edge)

    def get_node(self, node_id: str) -> NodeData | None:
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[NodeData]:
        result: list[NodeData] = []
        for edge in self._edges:
            if (
                direction in ("outgoing", "both")
                and edge.source_id == node_id
                and (edge_type is None or edge.edge_type == edge_type)
            ):
                node = self._nodes.get(edge.target_id)
                if node is not None:
                    result.append(node)
            if (
                direction in ("incoming", "both")
                and edge.target_id == node_id
                and (edge_type is None or edge.edge_type == edge_type)
            ):
                node = self._nodes.get(edge.source_id)
                if node is not None:
                    result.append(node)
        return result

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> TraversalResult:
        return TraversalResult(nodes=[], edges=[])

    def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        return []

    def delete_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._edges = [e for e in self._edges if e.source_id != node_id and e.target_id != node_id]

    def delete_edges_for_file(self, file_path: str) -> None:
        pass

    def get_file_hashes(self) -> dict[str, str]:
        return {}

    def get_nodes_for_file(self, file_path: str) -> list[NodeData]:
        return []

    def get_chunk_ids_for_file(self, file_path: str) -> list[str]:
        return []

    def delete_nodes_for_file(self, file_path: str) -> None:
        pass

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    def close(self) -> None:
        pass


# ===================================================================
# Integration Tests
# ===================================================================


class TestIntegration:
    """Integration tests using FakeGraph."""

    def test_full_rule_lifecycle(self) -> None:
        """Add, get, update, delete a rule through the full lifecycle."""
        graph = FakeGraph()
        mgr = RulesManager(graph)

        # Add
        rule = mgr.add_rule("Use type hints everywhere", scope="project")
        assert rule.content == "Use type hints everywhere"
        assert rule.scope == "project"

        # Get
        fetched = mgr.get_rule_by_id(rule.id)
        assert fetched is not None
        assert fetched.content == rule.content

        # Update
        updated = mgr.update_rule(rule.id, content="Always use type hints")
        assert updated is not None
        assert updated.content == "Always use type hints"

        # Delete
        deleted = mgr.delete_rule(rule.id)
        assert deleted is True
        assert mgr.get_rule_by_id(rule.id) is None

    def test_decision_with_alternatives(self) -> None:
        """Add decision + 2 alternatives, retrieve with get_decision_with_alternatives."""
        graph = FakeGraph()
        mgr = DecisionsManager(graph)

        # Add decision
        decision = mgr.add_decision(
            title="Use PostgreSQL",
            reasoning="Better JSON support",
            status="accepted",
        )

        # Add alternatives
        mgr.add_alternative(decision.id, "MySQL", reason_rejected="Weaker JSON support")
        mgr.add_alternative(decision.id, "SQLite", reason_rejected="Not suitable for production")

        # Retrieve
        dec, alternatives = mgr.get_decision_with_alternatives(decision.id)
        assert dec is not None
        assert dec.title == "Use PostgreSQL"
        assert len(alternatives) == 2
        alt_titles = {a.title for a in alternatives}
        assert alt_titles == {"MySQL", "SQLite"}

    def test_convention_with_governs_edge(self) -> None:
        """Add convention + module node, link via GOVERNS, verify neighbors."""
        graph = FakeGraph()
        conv_mgr = ConventionManager(graph)

        # Add convention
        conv = conv_mgr.add_convention(
            pattern="snake_case for functions",
            example="def my_function():",
            scope="project",
        )

        # Add a module node directly
        module_node = NodeData(
            id="mod-1",
            node_type="Module",
            properties={"name": "utils", "path": "nemesis/utils.py"},
        )
        graph.add_node(module_node)

        # Link convention to module
        conv_mgr.link_governs(conv.id, "mod-1")

        # Verify neighbors
        neighbors = graph.get_neighbors(conv.id, edge_type="GOVERNS")
        assert len(neighbors) == 1
        assert neighbors[0].id == "mod-1"
        assert neighbors[0].properties["name"] == "utils"

    def test_auto_learn_creates_graph_nodes(self) -> None:
        """process_message with 2 patterns creates 2 nodes in graph."""
        graph = FakeGraph()
        text = "Always use type hints. We decided to use FastAPI."
        results = process_message(text, graph)

        assert len(results) == 2

        # Verify types
        types = {r["type"] for r in results}
        assert types == {"rule", "decision"}

        # Verify nodes are in the graph
        for result in results:
            model = result["model"]
            node = graph.get_node(model.id)
            assert node is not None

    def test_session_context_with_memory(self) -> None:
        """SessionContext with rules query and result."""
        ctx = SessionContext()

        # Simulate a rules query
        ctx.add_query("rules")
        ctx.add_result(
            "rules",
            {
                "count": 2,
                "files": ["nemesis/memory/rules.py"],
            },
        )

        queries = ctx.get_queries()
        assert queries == ["rules"]

        results = ctx.get_results()
        assert len(results) == 1
        assert results[0]["query"] == "rules"
        assert results[0]["data"]["count"] == 2

        summary = ctx.build_summary()
        assert "rules" in summary
        assert "nemesis/memory/rules.py" in summary
