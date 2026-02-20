# Memory System — DecisionsManager (Basic + Advanced)

> **Arbeitspaket G3** — Teil 3 von 5 des Memory System Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** DecisionsManager mit CRUD, Alternatives, CHOSE/REJECTED-Edges und Cascade-Delete (Tasks 4 und 5).

**Tech Stack:** Python 3.11+, Pydantic, Graph Adapter (from 03), pytest
**Depends on:** [03-graph-layer](03-graph-layer.md), [06-mcp-server](06-mcp-server.md), [07a-memory-models.md](07a-memory-models.md)

**Tasks in diesem Paket:** 2 (Tasks 4–5 von 10)

---

## Task 4: DecisionsManager Basics

**Files:**
- `nemesis/memory/decisions.py`
- `tests/test_memory/test_decisions.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_decisions.py
"""Tests for DecisionsManager basic operations."""
import pytest
from unittest.mock import MagicMock
from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.models import DecisionModel, AlternativeModel
from nemesis.graph.adapter import NodeData


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return DecisionsManager(mock_graph)


class TestAddDecision:
    def test_add_decision(self, manager, mock_graph):
        dec = manager.add_decision("Use JWT", reasoning="Stateless")
        assert isinstance(dec, DecisionModel)
        assert dec.title == "Use JWT"
        mock_graph.add_node.assert_called_once()
        node = mock_graph.add_node.call_args[0][0]
        assert node.node_type == "Decision"

    def test_add_decision_default_status(self, manager):
        dec = manager.add_decision("X")
        assert dec.status == "proposed"


class TestGetDecisions:
    def test_get_decisions(self, manager, mock_graph):
        mock_graph.query.return_value = [
            {"id": "d1", "title": "JWT", "reasoning": "Stateless",
             "created_at": "2026-01-01T00:00:00+00:00", "status": "accepted"},
        ]
        decs = manager.get_decisions()
        assert len(decs) == 1
        assert decs[0].title == "JWT"


class TestAddAlternative:
    def test_add_alternative(self, manager, mock_graph):
        alt = manager.add_alternative("d1", "Sessions", "Stateful")
        assert isinstance(alt, AlternativeModel)
        assert alt.title == "Sessions"
        # Should create node + REJECTED edge
        assert mock_graph.add_node.call_count == 1
        assert mock_graph.add_edge.call_count == 1
        edge = mock_graph.add_edge.call_args[0][0]
        assert edge.edge_type == "REJECTED"
        assert edge.source_id == "d1"

    def test_add_alternative_chose_edge(self, manager, mock_graph):
        manager.add_chose_link("d1", "conv-1")
        mock_graph.add_edge.assert_called_once()
        edge = mock_graph.add_edge.call_args[0][0]
        assert edge.edge_type == "CHOSE"
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_decisions.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/decisions.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/decisions.py
"""Decisions manager — CRUD for architecture decisions in the graph."""
from __future__ import annotations

from typing import Literal

from nemesis.graph.adapter import GraphAdapter, NodeData, EdgeData
from nemesis.memory.models import DecisionModel, AlternativeModel


class DecisionsManager:
    """Manages Decision and Alternative nodes in the graph."""

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    def add_decision(self, title: str, reasoning: str = "",
                     status: str = "proposed") -> DecisionModel:
        dec = DecisionModel(title=title, reasoning=reasoning, status=status)
        self._graph.add_node(NodeData(
            id=dec.id, node_type="Decision",
            properties={
                "title": dec.title, "reasoning": dec.reasoning,
                "created_at": dec.created_at.isoformat(), "status": dec.status,
            },
        ))
        return dec

    def get_decisions(self) -> list[DecisionModel]:
        rows = self._graph.query("MATCH (d:Decision) RETURN d.*")
        return [DecisionModel(
            id=r["id"], title=r["title"], reasoning=r["reasoning"],
            created_at=r["created_at"], status=r["status"],
        ) for r in rows]

    def add_alternative(self, decision_id: str, title: str,
                        reason_rejected: str = "") -> AlternativeModel:
        alt = AlternativeModel(title=title, reason_rejected=reason_rejected)
        self._graph.add_node(NodeData(
            id=alt.id, node_type="Alternative",
            properties={"title": alt.title, "reason_rejected": alt.reason_rejected},
        ))
        self._graph.add_edge(EdgeData(
            source_id=decision_id, target_id=alt.id, edge_type="REJECTED",
        ))
        return alt

    def add_chose_link(self, decision_id: str, target_id: str) -> None:
        self._graph.add_edge(EdgeData(
            source_id=decision_id, target_id=target_id, edge_type="CHOSE",
        ))

    def update_decision_status(self, decision_id: str,
                               status: str) -> DecisionModel | None:
        node = self._graph.get_node(decision_id)
        if node is None or node.node_type != "Decision":
            return None
        p = node.properties
        dec = DecisionModel(
            id=node.id, title=p["title"], reasoning=p["reasoning"],
            created_at=p["created_at"], status=status,
        )
        self._graph.add_node(NodeData(
            id=dec.id, node_type="Decision",
            properties={
                "title": dec.title, "reasoning": dec.reasoning,
                "created_at": dec.created_at.isoformat(), "status": dec.status,
            },
        ))
        return dec

    def get_decision_with_alternatives(self, decision_id: str
                                       ) -> tuple[DecisionModel | None, list[AlternativeModel]]:
        node = self._graph.get_node(decision_id)
        if node is None or node.node_type != "Decision":
            return None, []
        p = node.properties
        dec = DecisionModel(
            id=node.id, title=p["title"], reasoning=p["reasoning"],
            created_at=p["created_at"], status=p["status"],
        )
        neighbors = self._graph.get_neighbors(decision_id, edge_type="REJECTED", direction="outgoing")
        alts = [AlternativeModel(
            id=n.id, title=n.properties["title"],
            reason_rejected=n.properties.get("reason_rejected", ""),
        ) for n in neighbors if n.node_type == "Alternative"]
        return dec, alts

    def delete_decision(self, decision_id: str) -> bool:
        node = self._graph.get_node(decision_id)
        if node is None or node.node_type != "Decision":
            return False
        neighbors = self._graph.get_neighbors(decision_id, edge_type="REJECTED", direction="outgoing")
        for n in neighbors:
            self._graph.delete_node(n.id)
        self._graph.delete_node(decision_id)
        return True
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_decisions.py -x -v
```

Expected: 5 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/decisions.py tests/test_memory/test_decisions.py
git commit -m "feat(memory): add DecisionsManager with CRUD and alternative linking

TDD Task 4/10 of 07-memory-system plan.
add_decision, get_decisions, add_alternative, add_chose_link
with REJECTED and CHOSE edges."
```

---

## Task 5: DecisionsManager Advanced

**Files:**
- `tests/test_memory/test_decisions.py` (update)

### Step 1 — Write failing test

Append to `tests/test_memory/test_decisions.py`:

```python
class TestUpdateDecisionStatus:
    def test_update_status(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="d1", node_type="Decision",
            properties={"title": "JWT", "reasoning": "Stateless",
                        "created_at": "2026-01-01T00:00:00+00:00", "status": "proposed"},
        )
        dec = manager.update_decision_status("d1", "accepted")
        assert dec is not None
        assert dec.status == "accepted"

    def test_update_nonexistent_returns_none(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        assert manager.update_decision_status("nope", "accepted") is None


class TestGetDecisionWithAlternatives:
    def test_get_with_alternatives(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="d1", node_type="Decision",
            properties={"title": "JWT", "reasoning": "R",
                        "created_at": "2026-01-01T00:00:00+00:00", "status": "accepted"},
        )
        mock_graph.get_neighbors.return_value = [
            NodeData(id="a1", node_type="Alternative",
                     properties={"title": "Sessions", "reason_rejected": "Stateful"}),
        ]
        dec, alts = manager.get_decision_with_alternatives("d1")
        assert dec is not None
        assert len(alts) == 1
        assert alts[0].title == "Sessions"


class TestDeleteDecision:
    def test_delete_with_alternatives(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="d1", node_type="Decision",
            properties={"title": "JWT", "reasoning": "R",
                        "created_at": "2026-01-01T00:00:00+00:00", "status": "accepted"},
        )
        mock_graph.get_neighbors.return_value = [
            NodeData(id="a1", node_type="Alternative",
                     properties={"title": "X", "reason_rejected": "Y"}),
        ]
        assert manager.delete_decision("d1") is True
        assert mock_graph.delete_node.call_count == 2  # alt + decision
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_decisions.py -x -v 2>&1 | head -30
```

Should PASS — implementation included in Task 4.

### Step 3 — Implementation

Already implemented in Task 4.

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_decisions.py -x -v
```

Expected: 9 tests PASSED (5 from Task 4 + 4 new).

### Step 5 — Commit

```bash
git add tests/test_memory/test_decisions.py
git commit -m "test(memory): add advanced DecisionsManager tests

TDD Task 5/10 of 07-memory-system plan.
Tests for status updates, decision+alternatives retrieval, cascade delete."
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 4 | DecisionsManager basics | `decisions.py` | 5 |
| 5 | DecisionsManager advanced | `test_decisions.py` | 4 |

---

**Navigation:**
- Vorheriges Paket: [07b-rules-manager.md](07b-rules-manager.md) (G2 — RulesManager)
- Nächstes Paket: [07d-conventions-context.md](07d-conventions-context.md) (G4 — ConventionManager + SessionContext)
