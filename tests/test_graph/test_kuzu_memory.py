"""Tests for memory-related node operations (Rules, Decisions, Conventions)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemesis.graph.adapter import EdgeData, NodeData
from nemesis.graph.kuzu import KuzuAdapter

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class TestKuzuMemoryNodes:
    """Tests for memory node types used by the memory system."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> Generator[KuzuAdapter, None, None]:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    def test_add_rule_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(
            NodeData(
                id="rule-001",
                node_type="Rule",
                properties={
                    "content": "Always use parameterized queries",
                    "scope": "global",
                    "created_at": "2026-02-20T10:00:00Z",
                    "source": "developer",
                },
            )
        )
        node = adapter.get_node("rule-001")
        assert node is not None
        assert node.properties["content"] == "Always use parameterized queries"
        assert node.properties["scope"] == "global"

    def test_add_decision_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(
            NodeData(
                id="dec-001",
                node_type="Decision",
                properties={
                    "title": "Use Kuzu as default graph DB",
                    "reasoning": "Zero-config, embedded, fast enough",
                    "created_at": "2026-02-20T10:00:00Z",
                    "status": "accepted",
                },
            )
        )
        node = adapter.get_node("dec-001")
        assert node is not None
        assert node.properties["title"] == "Use Kuzu as default graph DB"

    def test_add_alternative_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(
            NodeData(
                id="alt-001",
                node_type="Alternative",
                properties={
                    "title": "Use SQLite with virtual tables",
                    "reason_rejected": "No native graph traversal",
                },
            )
        )
        node = adapter.get_node("alt-001")
        assert node is not None
        assert node.properties["reason_rejected"] == "No native graph traversal"

    def test_add_convention_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(
            NodeData(
                id="conv-001",
                node_type="Convention",
                properties={
                    "pattern": "Use dataclasses for simple DTOs",
                    "example": "@dataclass\nclass UserDTO: ...",
                    "scope": "python",
                },
            )
        )
        node = adapter.get_node("conv-001")
        assert node is not None
        assert node.properties["pattern"] == "Use dataclasses for simple DTOs"

    def test_rule_applies_to_file(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(
            NodeData(
                id="rule-sql",
                node_type="Rule",
                properties={
                    "content": "Use parameterized queries",
                    "scope": "global",
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="file-db",
                node_type="File",
                properties={"path": "db.py", "language": "python"},
            )
        )
        adapter.add_edge(
            EdgeData(
                source_id="rule-sql",
                target_id="file-db",
                edge_type="APPLIES_TO",
            )
        )
        neighbors = adapter.get_neighbors("rule-sql", edge_type="APPLIES_TO", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "file-db" in ids

    def test_decision_chose_convention(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(
            NodeData(
                id="dec-dc",
                node_type="Decision",
                properties={"title": "Dataclass convention", "status": "accepted"},
            )
        )
        adapter.add_node(
            NodeData(
                id="conv-dc",
                node_type="Convention",
                properties={"pattern": "Use dataclasses"},
            )
        )
        adapter.add_edge(
            EdgeData(
                source_id="dec-dc",
                target_id="conv-dc",
                edge_type="CHOSE",
            )
        )
        neighbors = adapter.get_neighbors("dec-dc", edge_type="CHOSE", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "conv-dc" in ids

    def test_decision_rejected_alternative(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(
            NodeData(
                id="dec-rej",
                node_type="Decision",
                properties={"title": "Choose DB", "status": "accepted"},
            )
        )
        adapter.add_node(
            NodeData(
                id="alt-sqlite",
                node_type="Alternative",
                properties={"title": "SQLite"},
            )
        )
        adapter.add_edge(
            EdgeData(
                source_id="dec-rej",
                target_id="alt-sqlite",
                edge_type="REJECTED",
            )
        )
        neighbors = adapter.get_neighbors("dec-rej", edge_type="REJECTED", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "alt-sqlite" in ids

    def test_project_has_file(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(
            NodeData(
                id="proj-nemesis",
                node_type="Project",
                properties={"name": "nemesis", "root_path": "/home/user/nemesis"},
            )
        )
        adapter.add_node(
            NodeData(
                id="file-init",
                node_type="File",
                properties={"path": "__init__.py", "language": "python"},
            )
        )
        adapter.add_edge(
            EdgeData(
                source_id="proj-nemesis",
                target_id="file-init",
                edge_type="HAS_FILE",
            )
        )
        neighbors = adapter.get_neighbors(
            "proj-nemesis", edge_type="HAS_FILE", direction="outgoing"
        )
        ids = {n.id for n in neighbors}
        assert "file-init" in ids
