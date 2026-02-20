# Memory System — RulesManager (Basic + Advanced)

> **Arbeitspaket G2** — Teil 2 von 5 des Memory System Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** RulesManager mit CRUD-Operationen, Scope-Filterung und APPLIES_TO-Edges (Tasks 2 und 3).

**Tech Stack:** Python 3.11+, Pydantic, Graph Adapter (from 03), pytest
**Depends on:** [03-graph-layer](03-graph-layer.md), [06-mcp-server](06-mcp-server.md), [07a-memory-models.md](07a-memory-models.md)

**Tasks in diesem Paket:** 2 (Tasks 2–3 von 10)

---

## Task 2: RulesManager Basics

**Files:**
- `nemesis/memory/rules.py`
- `tests/test_memory/test_rules.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_rules.py
"""Tests for RulesManager basic CRUD."""
import pytest
from unittest.mock import MagicMock
from nemesis.memory.rules import RulesManager
from nemesis.memory.models import RuleModel
from nemesis.graph.adapter import NodeData


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return RulesManager(mock_graph)


class TestAddRule:
    def test_add_rule_calls_add_node(self, manager, mock_graph):
        rule = manager.add_rule("Always use type hints", scope="project")
        assert isinstance(rule, RuleModel)
        mock_graph.add_node.assert_called_once()
        node = mock_graph.add_node.call_args[0][0]
        assert node.node_type == "Rule"
        assert node.properties["content"] == "Always use type hints"

    def test_add_rule_returns_model(self, manager):
        rule = manager.add_rule("Use pytest", source="auto")
        assert rule.content == "Use pytest"
        assert rule.source == "auto"


class TestGetRules:
    def test_get_rules_queries_graph(self, manager, mock_graph):
        mock_graph.query.return_value = [
            {"id": "r1", "content": "Rule A", "scope": "project",
             "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        ]
        rules = manager.get_rules()
        assert len(rules) == 1
        assert rules[0].content == "Rule A"

    def test_get_rules_empty(self, manager):
        rules = manager.get_rules()
        assert rules == []


class TestGetRuleById:
    def test_get_existing_rule(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="r1", node_type="Rule",
            properties={"content": "X", "scope": "project",
                        "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        )
        rule = manager.get_rule_by_id("r1")
        assert rule is not None
        assert rule.content == "X"

    def test_get_nonexistent_rule(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        assert manager.get_rule_by_id("nope") is None
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_rules.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/rules.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/rules.py
"""Rules manager — CRUD for project rules in the graph."""
from __future__ import annotations

from nemesis.graph.adapter import GraphAdapter, NodeData, EdgeData
from nemesis.memory.models import RuleModel


class RulesManager:
    """Manages Rule nodes in the graph."""

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    def add_rule(self, content: str, scope: str = "project",
                 source: str = "manual") -> RuleModel:
        rule = RuleModel(content=content, scope=scope, source=source)
        self._graph.add_node(NodeData(
            id=rule.id, node_type="Rule",
            properties={
                "content": rule.content, "scope": rule.scope,
                "created_at": rule.created_at.isoformat(), "source": rule.source,
            },
        ))
        return rule

    def get_rules(self) -> list[RuleModel]:
        rows = self._graph.query("MATCH (r:Rule) RETURN r.*")
        return [RuleModel(
            id=r["id"], content=r["content"], scope=r["scope"],
            created_at=r["created_at"], source=r["source"],
        ) for r in rows]

    def get_rule_by_id(self, rule_id: str) -> RuleModel | None:
        node = self._graph.get_node(rule_id)
        if node is None or node.node_type != "Rule":
            return None
        p = node.properties
        return RuleModel(
            id=node.id, content=p["content"], scope=p["scope"],
            created_at=p["created_at"], source=p["source"],
        )

    def update_rule(self, rule_id: str, **kwargs: object) -> RuleModel | None:
        existing = self.get_rule_by_id(rule_id)
        if existing is None:
            return None
        updated = existing.model_copy(update=kwargs)
        self._graph.add_node(NodeData(
            id=updated.id, node_type="Rule",
            properties={
                "content": updated.content, "scope": updated.scope,
                "created_at": updated.created_at.isoformat(), "source": updated.source,
            },
        ))
        return updated

    def delete_rule(self, rule_id: str) -> bool:
        if self.get_rule_by_id(rule_id) is None:
            return False
        self._graph.delete_node(rule_id)
        return True

    def get_rules_for_scope(self, scope: str) -> list[RuleModel]:
        rows = self._graph.query(
            "MATCH (r:Rule) WHERE r.scope = $scope RETURN r.*",
            parameters={"scope": scope},
        )
        return [RuleModel(
            id=r["id"], content=r["content"], scope=r["scope"],
            created_at=r["created_at"], source=r["source"],
        ) for r in rows]

    def link_rule_to_target(self, rule_id: str, target_id: str) -> None:
        self._graph.add_edge(EdgeData(
            source_id=rule_id, target_id=target_id, edge_type="APPLIES_TO",
        ))
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_rules.py -x -v
```

Expected: 5 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/rules.py tests/test_memory/test_rules.py
git commit -m "feat(memory): add RulesManager with basic CRUD operations

TDD Task 2/10 of 07-memory-system plan.
add_rule, get_rules, get_rule_by_id using graph adapter."
```

---

## Task 3: RulesManager Advanced

**Files:**
- `tests/test_memory/test_rules.py` (update)

### Step 1 — Write failing test

Append to `tests/test_memory/test_rules.py`:

```python
class TestUpdateRule:
    def test_update_existing_rule(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="r1", node_type="Rule",
            properties={"content": "Old", "scope": "project",
                        "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        )
        updated = manager.update_rule("r1", content="New content")
        assert updated is not None
        assert updated.content == "New content"
        assert mock_graph.add_node.call_count == 1

    def test_update_nonexistent_returns_none(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        assert manager.update_rule("nope", content="X") is None


class TestDeleteRule:
    def test_delete_existing_rule(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="r1", node_type="Rule",
            properties={"content": "X", "scope": "project",
                        "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        )
        assert manager.delete_rule("r1") is True
        mock_graph.delete_node.assert_called_once_with("r1")

    def test_delete_nonexistent_returns_false(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        assert manager.delete_rule("nope") is False


class TestRulesForScope:
    def test_get_rules_for_scope(self, manager, mock_graph):
        mock_graph.query.return_value = [
            {"id": "r1", "content": "A", "scope": "module",
             "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        ]
        rules = manager.get_rules_for_scope("module")
        assert len(rules) == 1
        assert rules[0].scope == "module"
        mock_graph.query.assert_called_once()


class TestLinkRule:
    def test_link_rule_to_target(self, manager, mock_graph):
        manager.link_rule_to_target("r1", "file-001")
        mock_graph.add_edge.assert_called_once()
        edge = mock_graph.add_edge.call_args[0][0]
        assert edge.edge_type == "APPLIES_TO"
        assert edge.source_id == "r1"
        assert edge.target_id == "file-001"
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_rules.py -x -v 2>&1 | head -30
```

Tests should PASS immediately — implementation was included in Task 2.

### Step 3 — Implementation

Already implemented in Task 2 (`update_rule`, `delete_rule`, `get_rules_for_scope`, `link_rule_to_target`).

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_rules.py -x -v
```

Expected: 10 tests PASSED (5 from Task 2 + 5 new).

### Step 5 — Commit

```bash
git add tests/test_memory/test_rules.py
git commit -m "test(memory): add advanced RulesManager tests

TDD Task 3/10 of 07-memory-system plan.
Tests for update, delete, scope filtering, and APPLIES_TO edges."
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 2 | RulesManager basics | `rules.py` | 5 |
| 3 | RulesManager advanced | `test_rules.py` | 5 |

---

**Navigation:**
- Vorheriges Paket: [07a-memory-models.md](07a-memory-models.md) (G1 — Pydantic Data Models)
- Nächstes Paket: [07c-decisions-manager.md](07c-decisions-manager.md) (G3 — DecisionsManager)
