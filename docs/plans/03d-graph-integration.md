# Graph Layer — Arbeitspaket C4: Graph Integration

> **Arbeitspaket C4** — Teil 4 von 4 des Graph Layer Plans

**Goal:** Factory-Funktion fuer Adapter-Erstellung und vollstaendige End-to-End Integration Tests (Tasks 8, 10).

**Architecture:** `create_graph_adapter()` Factory-Funktion erstellt den passenden Adapter (Kuzu oder Neo4j) basierend auf Konfiguration. Integration Tests pruefen den kompletten Workflow mit echtem Kuzu-Backend.

**Tech Stack:** Python 3.11+, Kuzu >= 0.4, pytest

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [03a-graph-foundation](03a-graph-foundation.md) (C1), [03b-kuzu-crud](03b-kuzu-crud.md) (C2), [03c-neo4j-memory](03c-neo4j-memory.md) (C3)

---

## Task 8: Graph Module Factory & __init__.py

**Files:**
- `tests/test_graph/test_factory.py`
- `nemesis/graph/__init__.py` (replace)

### Step 1 — Write failing test

```python
# tests/test_graph/test_factory.py
"""Tests for the graph module factory function."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from nemesis.graph import create_graph_adapter
from nemesis.graph.adapter import GraphAdapter


class TestCreateGraphAdapter:
    """Tests for the factory function."""

    def test_create_kuzu_adapter(self, tmp_path: Path) -> None:
        adapter = create_graph_adapter(
            backend="kuzu",
            db_path=str(tmp_path / "test_graph"),
        )
        assert isinstance(adapter, GraphAdapter)
        adapter.close()

    def test_create_kuzu_is_default(self, tmp_path: Path) -> None:
        adapter = create_graph_adapter(
            db_path=str(tmp_path / "test_graph"),
        )
        from nemesis.graph.kuzu import KuzuAdapter
        assert isinstance(adapter, KuzuAdapter)
        adapter.close()

    def test_create_neo4j_adapter(self) -> None:
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_driver.GraphDatabase.driver.return_value = MagicMock()
            adapter = create_graph_adapter(
                backend="neo4j",
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test",
            )
            assert isinstance(adapter, GraphAdapter)
            adapter.close()

    def test_create_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown graph backend"):
            create_graph_adapter(backend="invalid")

    def test_kuzu_adapter_creates_schema(self, tmp_path: Path) -> None:
        adapter = create_graph_adapter(
            backend="kuzu",
            db_path=str(tmp_path / "test_graph"),
            create_schema=True,
        )
        # Schema should be created, so we can add nodes
        from nemesis.graph.adapter import NodeData
        adapter.add_node(NodeData(id="test", node_type="File", properties={"path": "test.py"}))
        assert adapter.get_node("test") is not None
        adapter.close()
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_factory.py -x -v 2>&1 | head -20
```

Tests fail because `create_graph_adapter` does not exist.

### Step 3 — Implement

```python
# nemesis/graph/__init__.py — REPLACE entire file
"""Graph module — abstract adapter, Kuzu (default), Neo4j (optional).

This module provides:
- GraphAdapter protocol defining the common graph interface
- KuzuAdapter for embedded graph storage (default, zero-config)
- Neo4jAdapter for remote graph database (optional)
- Factory function for creating graph adapters from config
"""
from __future__ import annotations

from typing import Any

from nemesis.graph.adapter import (
    EDGE_TYPES,
    NODE_SCHEMAS,
    NODE_TYPES,
    EdgeData,
    GraphAdapter,
    NodeData,
    TraversalResult,
)


def create_graph_adapter(
    backend: str = "kuzu",
    create_schema: bool = False,
    **kwargs: Any,
) -> GraphAdapter:
    """Factory function to create a graph adapter.

    Args:
        backend: Either "kuzu" (default) or "neo4j".
        create_schema: If True, call create_schema() after creation.
        **kwargs: Passed to the adapter constructor.
            For "kuzu": db_path (required).
            For "neo4j": uri, user, password, database.

    Returns:
        A GraphAdapter instance.

    Raises:
        ValueError: If backend is not recognized.
    """
    if backend == "kuzu":
        from nemesis.graph.kuzu import KuzuAdapter

        adapter = KuzuAdapter(**kwargs)
    elif backend == "neo4j":
        from nemesis.graph.neo4j import Neo4jAdapter

        adapter = Neo4jAdapter(**kwargs)
    else:
        raise ValueError(
            f"Unknown graph backend: {backend!r}. Supported: 'kuzu', 'neo4j'"
        )

    if create_schema:
        adapter.create_schema()

    return adapter


__all__ = [
    "EDGE_TYPES",
    "NODE_SCHEMAS",
    "NODE_TYPES",
    "EdgeData",
    "GraphAdapter",
    "NodeData",
    "TraversalResult",
    "create_graph_adapter",
]
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_factory.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/graph/__init__.py tests/test_graph/test_factory.py
git commit -m "feat(graph): add create_graph_adapter factory function

TDD Task 8/10 of 03-graph-layer plan.
Factory creates KuzuAdapter (default) or Neo4jAdapter based on
backend parameter. Optional create_schema flag for convenience.
Module __init__ re-exports all public API."
```

---

## Task 10: Integration Test — Full Graph Workflow

**Files:**
- `tests/test_graph/test_integration.py`

### Step 1 — Write failing test

```python
# tests/test_graph/test_integration.py
"""Integration tests for the graph module.

Tests the full workflow: create adapter via factory, populate a
realistic code graph, query it, perform delta updates.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from nemesis.graph import create_graph_adapter
from nemesis.graph.adapter import NodeData, EdgeData, TraversalResult


class TestGraphIntegration:
    """End-to-end tests with a real Kuzu database."""

    @pytest.fixture
    def adapter(self, tmp_path: Path):
        a = create_graph_adapter(
            backend="kuzu",
            db_path=str(tmp_path / "integration_graph"),
            create_schema=True,
        )
        yield a
        a.close()

    def test_index_and_query_file(self, adapter) -> None:
        """Simulate indexing a Python file and querying its structure."""
        # Add a file with classes and functions
        adapter.add_node(NodeData(id="file-svc", node_type="File",
            properties={"path": "service.py", "language": "python", "hash": "abc123", "size": 2048}))
        adapter.add_node(NodeData(id="class-userservice", node_type="Class",
            properties={"name": "UserService", "file": "service.py", "line_start": 5, "line_end": 80}))
        adapter.add_node(NodeData(id="method-create", node_type="Method",
            properties={"name": "create_user", "class_name": "UserService", "file": "service.py",
                         "line_start": 10, "line_end": 25, "visibility": "public"}))
        adapter.add_node(NodeData(id="method-delete", node_type="Method",
            properties={"name": "delete_user", "class_name": "UserService", "file": "service.py",
                         "line_start": 27, "line_end": 40, "visibility": "public"}))
        adapter.add_node(NodeData(id="func-validate", node_type="Function",
            properties={"name": "validate_email", "file": "service.py", "line_start": 82, "line_end": 90}))

        # Edges
        adapter.add_edge(EdgeData(source_id="file-svc", target_id="class-userservice", edge_type="CONTAINS"))
        adapter.add_edge(EdgeData(source_id="file-svc", target_id="func-validate", edge_type="CONTAINS"))
        adapter.add_edge(EdgeData(source_id="class-userservice", target_id="method-create", edge_type="HAS_METHOD"))
        adapter.add_edge(EdgeData(source_id="class-userservice", target_id="method-delete", edge_type="HAS_METHOD"))

        # Query: What does service.py contain?
        contents = adapter.get_neighbors("file-svc", edge_type="CONTAINS", direction="outgoing")
        content_ids = {n.id for n in contents}
        assert "class-userservice" in content_ids
        assert "func-validate" in content_ids

        # Query: What methods does UserService have?
        methods = adapter.get_neighbors("class-userservice", edge_type="HAS_METHOD", direction="outgoing")
        method_names = {n.properties["name"] for n in methods}
        assert "create_user" in method_names
        assert "delete_user" in method_names

    def test_delta_update_workflow(self, adapter) -> None:
        """Simulate a delta update: delete old data, add new."""
        # Initial index
        adapter.add_node(NodeData(id="file-old", node_type="File",
            properties={"path": "changing.py", "language": "python", "hash": "old_hash"}))
        adapter.add_node(NodeData(id="func-old", node_type="Function",
            properties={"name": "old_func", "file": "changing.py"}))
        adapter.add_edge(EdgeData(source_id="file-old", target_id="func-old", edge_type="CONTAINS"))

        # Verify initial state
        assert adapter.get_node("func-old") is not None
        hashes = adapter.get_file_hashes()
        assert hashes.get("changing.py") == "old_hash"

        # Delta update: file changed
        adapter.delete_nodes_for_file("changing.py")

        # Old data gone
        assert adapter.get_node("file-old") is None
        assert adapter.get_node("func-old") is None

        # Add new version
        adapter.add_node(NodeData(id="file-new", node_type="File",
            properties={"path": "changing.py", "language": "python", "hash": "new_hash"}))
        adapter.add_node(NodeData(id="func-new", node_type="Function",
            properties={"name": "new_func", "file": "changing.py"}))
        adapter.add_edge(EdgeData(source_id="file-new", target_id="func-new", edge_type="CONTAINS"))

        # New data present
        hashes = adapter.get_file_hashes()
        assert hashes.get("changing.py") == "new_hash"
        nodes = adapter.get_nodes_for_file("changing.py")
        ids = {n.id for n in nodes}
        assert "file-new" in ids
        assert "func-new" in ids

    def test_traversal_with_realistic_graph(self, adapter) -> None:
        """Test traversal through a realistic dependency chain."""
        # auth.py -> AuthService.authenticate() -> calls -> validate_token() -> calls -> db_lookup()
        adapter.add_node(NodeData(id="f-auth", node_type="File", properties={"path": "auth.py"}))
        adapter.add_node(NodeData(id="fn-authenticate", node_type="Function",
            properties={"name": "authenticate", "file": "auth.py"}))
        adapter.add_node(NodeData(id="fn-validate", node_type="Function",
            properties={"name": "validate_token", "file": "auth.py"}))
        adapter.add_node(NodeData(id="fn-lookup", node_type="Function",
            properties={"name": "db_lookup", "file": "db.py"}))

        adapter.add_edge(EdgeData(source_id="f-auth", target_id="fn-authenticate", edge_type="CONTAINS"))
        adapter.add_edge(EdgeData(source_id="fn-authenticate", target_id="fn-validate", edge_type="CALLS"))
        adapter.add_edge(EdgeData(source_id="fn-validate", target_id="fn-lookup", edge_type="CALLS"))

        # Traverse from authenticate
        result = adapter.traverse("fn-authenticate", max_depth=2)
        ids = {n.id for n in result.nodes}
        assert "fn-authenticate" in ids
        assert "fn-validate" in ids
        assert "fn-lookup" in ids

    def test_memory_integration(self, adapter) -> None:
        """Test memory nodes alongside code nodes."""
        # Code node
        adapter.add_node(NodeData(id="file-db", node_type="File",
            properties={"path": "database.py", "language": "python"}))
        # Rule that applies to this file
        adapter.add_node(NodeData(id="rule-params", node_type="Rule",
            properties={"content": "Always use parameterized queries", "scope": "database"}))
        adapter.add_edge(EdgeData(source_id="rule-params", target_id="file-db", edge_type="APPLIES_TO"))

        # Query: What rules apply to database.py?
        rules = adapter.get_neighbors("file-db", edge_type="APPLIES_TO", direction="incoming")
        assert len(rules) >= 1
        rule_contents = {n.properties.get("content", "") for n in rules}
        assert "Always use parameterized queries" in rule_contents

    def test_chunk_lifecycle(self, adapter) -> None:
        """Test chunk creation and file-level deletion."""
        adapter.add_node(NodeData(id="f-chunk", node_type="File",
            properties={"path": "chunked.py", "language": "python"}))
        adapter.add_node(NodeData(id="fn-big", node_type="Function",
            properties={"name": "big_function", "file": "chunked.py"}))
        adapter.add_node(NodeData(id="chunk-1", node_type="Chunk",
            properties={"content": "part 1...", "token_count": 100, "parent_type": "Function"}))
        adapter.add_node(NodeData(id="chunk-2", node_type="Chunk",
            properties={"content": "part 2...", "token_count": 100, "parent_type": "Function"}))

        adapter.add_edge(EdgeData(source_id="f-chunk", target_id="fn-big", edge_type="CONTAINS"))
        adapter.add_edge(EdgeData(source_id="chunk-1", target_id="fn-big", edge_type="CHUNK_OF"))
        adapter.add_edge(EdgeData(source_id="chunk-2", target_id="fn-big", edge_type="CHUNK_OF"))

        # Get chunk IDs for file
        chunk_ids = adapter.get_chunk_ids_for_file("chunked.py")
        assert set(chunk_ids) == {"chunk-1", "chunk-2"}

        # Delete file data — should remove chunks too
        adapter.delete_nodes_for_file("chunked.py")
        assert adapter.get_node("chunk-1") is None
        assert adapter.get_node("chunk-2") is None
        assert adapter.get_node("fn-big") is None
        assert adapter.get_node("f-chunk") is None

    def test_clear_removes_all_data(self, adapter) -> None:
        """clear() removes every node and edge."""
        adapter.add_node(NodeData(id="n1", node_type="File", properties={"path": "a.py"}))
        adapter.add_node(NodeData(id="n2", node_type="Function", properties={"name": "f"}))
        adapter.add_edge(EdgeData(source_id="n1", target_id="n2", edge_type="CONTAINS"))

        adapter.clear()

        assert adapter.get_node("n1") is None
        assert adapter.get_node("n2") is None
        assert adapter.get_file_hashes() == {}
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_integration.py -x -v 2>&1 | head -20
```

### Step 3 — Implementation already complete in previous tasks

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_integration.py -x -v
```

### Step 5 — Commit

```bash
git add tests/test_graph/test_integration.py
git commit -m "test(graph): add full integration tests for graph module

TDD Task 10/10 of 03-graph-layer plan.
End-to-end tests: index file + query, delta update workflow,
realistic traversal, memory node integration, chunk lifecycle,
and clear(). All using real Kuzu database in temp directory."
```

---

## Summary

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 8 | Factory Function + Module Init | `__init__.py`, `test_factory.py` | 5 |
| 10 | Full Integration Tests | `test_integration.py` | 6 |
| **Gesamt C4** | | **1 Source + 2 Test-Dateien** | **~11 Tests** |

---

### Gesamtuebersicht aller Pakete

| Paket | Tasks | Dateien | Tests |
|-------|-------|---------|-------|
| [C1 — Graph Foundation](03a-graph-foundation.md) | 1, 2 | 2 Source + 2 Test | ~17 |
| [C2 — Kuzu CRUD](03b-kuzu-crud.md) | 3, 4, 5, 6 | 1 Test (erweitert) | ~31 |
| [C3 — Neo4j & Memory](03c-neo4j-memory.md) | 7, 9 | 1 Source + 2 Test | ~15 |
| [C4 — Graph Integration](03d-graph-integration.md) | 8, 10 | 1 Source + 2 Test | ~11 |
| **Gesamt** | **10 Tasks** | **4 Source + 6 Test-Dateien** | **~74 Tests** |

### Files created

```
nemesis/graph/
├── __init__.py         # Module init + create_graph_adapter() factory
├── adapter.py          # GraphAdapter Protocol, NodeData, EdgeData, TraversalResult, schema constants
├── kuzu.py             # KuzuAdapter — embedded graph (default)
└── neo4j.py            # Neo4jAdapter — remote graph (optional)

tests/test_graph/
├── __init__.py
├── test_adapter.py     # Protocol + data model tests
├── test_kuzu.py        # KuzuAdapter CRUD, edges, file ops, traversal
├── test_kuzu_memory.py # Memory node types (Rule, Decision, Convention)
├── test_neo4j.py       # Neo4jAdapter with mocked driver
├── test_factory.py     # create_graph_adapter() factory
└── test_integration.py # End-to-end graph workflows
```

### Dependencies

```toml
[project]
dependencies = [
    "kuzu>=0.4",
]

[project.optional-dependencies]
neo4j = ["neo4j>=5.0"]
```

---

**Navigation:**
- Vorheriges Paket: [C3 — Neo4j & Memory](03c-neo4j-memory.md)
- Nachstes Paket: --
- Gesamtplan: [03-graph-layer.md](03-graph-layer.md)
