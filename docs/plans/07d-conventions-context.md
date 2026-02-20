# Memory System — ConventionManager + SessionContext

> **Arbeitspaket G4** — Teil 4 von 5 des Memory System Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** ConventionManager mit GOVERNS-Edges und SessionContext fuer In-Memory Session-Tracking (Tasks 6 und 7).

**Tech Stack:** Python 3.11+, Pydantic, Graph Adapter (from 03), pytest
**Depends on:** [03-graph-layer](03-graph-layer.md), [06-mcp-server](06-mcp-server.md), [07a-memory-models.md](07a-memory-models.md)

**Tasks in diesem Paket:** 2 (Tasks 6–7 von 10)

---

## Task 6: ConventionManager

**Files:**
- `nemesis/memory/conventions.py`
- `tests/test_memory/test_conventions.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_conventions.py
"""Tests for ConventionManager."""
import pytest
from unittest.mock import MagicMock
from nemesis.memory.conventions import ConventionManager
from nemesis.memory.models import ConventionModel
from nemesis.graph.adapter import NodeData


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return ConventionManager(mock_graph)


def test_add_convention(manager, mock_graph):
    conv = manager.add_convention("snake_case", example="my_var", scope="project")
    assert isinstance(conv, ConventionModel)
    assert conv.pattern == "snake_case"
    mock_graph.add_node.assert_called_once()
    node = mock_graph.add_node.call_args[0][0]
    assert node.node_type == "Convention"


def test_get_conventions(manager, mock_graph):
    mock_graph.query.return_value = [
        {"id": "c1", "pattern": "snake_case", "example": "my_var",
         "scope": "project", "created_at": "2026-01-01T00:00:00+00:00"},
    ]
    convs = manager.get_conventions()
    assert len(convs) == 1
    assert convs[0].pattern == "snake_case"


def test_link_convention_governs(manager, mock_graph):
    manager.link_governs("c1", "module-001")
    mock_graph.add_edge.assert_called_once()
    edge = mock_graph.add_edge.call_args[0][0]
    assert edge.edge_type == "GOVERNS"
    assert edge.source_id == "c1"
    assert edge.target_id == "module-001"


def test_delete_convention(manager, mock_graph):
    mock_graph.get_node.return_value = NodeData(
        id="c1", node_type="Convention",
        properties={"pattern": "X", "example": "", "scope": "project",
                    "created_at": "2026-01-01T00:00:00+00:00"},
    )
    assert manager.delete_convention("c1") is True
    mock_graph.delete_node.assert_called_once_with("c1")
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_conventions.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/conventions.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/conventions.py
"""Convention manager — CRUD for coding conventions in the graph."""
from __future__ import annotations

from nemesis.graph.adapter import GraphAdapter, NodeData, EdgeData
from nemesis.memory.models import ConventionModel


class ConventionManager:
    """Manages Convention nodes in the graph."""

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    def add_convention(self, pattern: str, example: str = "",
                       scope: str = "project") -> ConventionModel:
        conv = ConventionModel(pattern=pattern, example=example, scope=scope)
        self._graph.add_node(NodeData(
            id=conv.id, node_type="Convention",
            properties={
                "pattern": conv.pattern, "example": conv.example,
                "scope": conv.scope, "created_at": conv.created_at.isoformat(),
            },
        ))
        return conv

    def get_conventions(self) -> list[ConventionModel]:
        rows = self._graph.query("MATCH (c:Convention) RETURN c.*")
        return [ConventionModel(
            id=r["id"], pattern=r["pattern"], example=r["example"],
            scope=r["scope"], created_at=r["created_at"],
        ) for r in rows]

    def link_governs(self, convention_id: str, target_id: str) -> None:
        self._graph.add_edge(EdgeData(
            source_id=convention_id, target_id=target_id, edge_type="GOVERNS",
        ))

    def delete_convention(self, convention_id: str) -> bool:
        node = self._graph.get_node(convention_id)
        if node is None or node.node_type != "Convention":
            return False
        self._graph.delete_node(convention_id)
        return True
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_conventions.py -x -v
```

Expected: 4 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/conventions.py tests/test_memory/test_conventions.py
git commit -m "feat(memory): add ConventionManager with GOVERNS edge support

TDD Task 6/10 of 07-memory-system plan.
add_convention, get_conventions, link_governs, delete_convention."
```

---

## Task 7: SessionContext

**Files:**
- `nemesis/memory/context.py`
- `tests/test_memory/test_context.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_context.py
"""Tests for SessionContext — in-memory session tracking."""
import pytest
from nemesis.memory.context import SessionContext


def test_empty_session():
    ctx = SessionContext()
    assert ctx.get_queries() == []
    assert ctx.get_results() == []


def test_add_query():
    ctx = SessionContext()
    ctx.add_query("What does auth do?")
    ctx.add_query("Show dependencies")
    assert len(ctx.get_queries()) == 2
    assert ctx.get_queries()[0] == "What does auth do?"


def test_add_result():
    ctx = SessionContext()
    ctx.add_result("query1", {"files": ["auth.py"], "symbols": ["login"]})
    results = ctx.get_results()
    assert len(results) == 1
    assert results[0]["query"] == "query1"
    assert results[0]["data"]["files"] == ["auth.py"]


def test_build_summary():
    ctx = SessionContext()
    ctx.add_query("auth flow")
    ctx.add_result("auth flow", {"files": ["auth.py"]})
    summary = ctx.build_summary()
    assert "auth flow" in summary
    assert "auth.py" in summary


def test_clear():
    ctx = SessionContext()
    ctx.add_query("test")
    ctx.add_result("test", {"x": 1})
    ctx.clear()
    assert ctx.get_queries() == []
    assert ctx.get_results() == []
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_context.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/context.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/context.py
"""Session context — tracks queries and results within a session."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionContext:
    """In-memory session context (not persisted to graph)."""

    _queries: list[str] = field(default_factory=list)
    _results: list[dict] = field(default_factory=list)

    def add_query(self, query: str) -> None:
        self._queries.append(query)

    def add_result(self, query: str, data: dict) -> None:
        self._results.append({"query": query, "data": data})

    def get_queries(self) -> list[str]:
        return list(self._queries)

    def get_results(self) -> list[dict]:
        return list(self._results)

    def build_summary(self) -> str:
        lines: list[str] = []
        for entry in self._results:
            q = entry["query"]
            data = entry["data"]
            files = data.get("files", [])
            symbols = data.get("symbols", [])
            parts = [f"Query: {q}"]
            if files:
                parts.append(f"  Files: {', '.join(files)}")
            if symbols:
                parts.append(f"  Symbols: {', '.join(symbols)}")
            lines.extend(parts)
        if not lines:
            for q in self._queries:
                lines.append(f"Query: {q}")
        return "\n".join(lines) if lines else ""

    def clear(self) -> None:
        self._queries.clear()
        self._results.clear()
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_context.py -x -v
```

Expected: 5 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/context.py tests/test_memory/test_context.py
git commit -m "feat(memory): add SessionContext for in-memory session tracking

TDD Task 7/10 of 07-memory-system plan.
Tracks queries, results, builds text summary. Not persisted to graph."
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 6 | ConventionManager | `conventions.py` | 4 |
| 7 | SessionContext | `context.py` | 5 |

---

**Navigation:**
- Vorheriges Paket: [07c-decisions-manager.md](07c-decisions-manager.md) (G3 — DecisionsManager)
- Nächstes Paket: [07e-auto-learn.md](07e-auto-learn.md) (G5 — AutoLearn + Integration)
