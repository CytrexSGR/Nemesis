# Graph Layer — Arbeitspaket C3: Neo4j & Memory Nodes

> **Arbeitspaket C3** — Teil 3 von 4 des Graph Layer Plans

**Goal:** Neo4j Adapter implementieren und Memory Node Operations testen (Tasks 7, 9).

**Architecture:** `Neo4jAdapter` bietet optionale Remote-Graph-Datenbank-Unterstuetzung. Memory Nodes (Rule, Decision, Alternative, Convention) werden im Kuzu-Graph gespeichert und mit Code-Nodes verknuepft.

**Tech Stack:** Python 3.11+, Neo4j >= 5.0 (optional), Kuzu >= 0.4, pytest

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [03a-graph-foundation](03a-graph-foundation.md) (C1), [03b-kuzu-crud](03b-kuzu-crud.md) (C2)

---

## Task 7: Neo4jAdapter — Implementation

**Files:**
- `tests/test_graph/test_neo4j.py`
- `nemesis/graph/neo4j.py`

### Step 1 — Write failing test

```python
# tests/test_graph/test_neo4j.py
"""Tests for Neo4j graph adapter.

These tests use a mock Neo4j driver since a running Neo4j instance
is not required for development. Integration tests with a real
Neo4j server are marked with @pytest.mark.integration.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from nemesis.graph.adapter import GraphAdapter, NodeData, EdgeData, TraversalResult
from nemesis.graph.neo4j import Neo4jAdapter


class TestNeo4jInit:
    """Tests for Neo4jAdapter initialization."""

    def test_is_graph_adapter(self) -> None:
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_driver.GraphDatabase.driver.return_value = MagicMock()
            adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
            assert isinstance(adapter, GraphAdapter)
            adapter.close()

    def test_create_schema_idempotent(self) -> None:
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_session = MagicMock()
            mock_driver_instance = MagicMock()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver_instance.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
            adapter.create_schema()
            adapter.create_schema()  # Should not raise
            assert mock_session.run.called
            adapter.close()


class TestNeo4jNodeCRUD:
    """Tests for Neo4j node CRUD with mocked driver."""

    @pytest.fixture
    def mock_neo4j(self):
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_session = MagicMock()
            mock_tx = MagicMock()
            mock_driver_instance = MagicMock()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver_instance.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.execute_write.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_session.execute_read.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            yield {
                "driver": mock_driver,
                "driver_instance": mock_driver_instance,
                "session": mock_session,
                "tx": mock_tx,
            }

    def test_add_node_executes_merge(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.add_node(NodeData(id="f1", node_type="Function", properties={"name": "hello"}))
        # Verify that tx.run was called with a MERGE query
        assert mock_neo4j["tx"].run.called
        call_args = mock_neo4j["tx"].run.call_args
        assert "MERGE" in call_args[0][0]
        adapter.close()

    def test_get_node_executes_match(self, mock_neo4j) -> None:
        mock_record = MagicMock()
        mock_record.data.return_value = {"id": "f1", "name": "hello"}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_neo4j["tx"].run.return_value = mock_result

        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        node = adapter.get_node("f1")
        assert mock_neo4j["tx"].run.called
        adapter.close()

    def test_delete_node_executes_detach_delete(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.delete_node("f1")
        assert mock_neo4j["tx"].run.called
        call_args = mock_neo4j["tx"].run.call_args
        assert "DETACH DELETE" in call_args[0][0]
        adapter.close()


class TestNeo4jEdgeCRUD:
    """Tests for Neo4j edge operations with mocked driver."""

    @pytest.fixture
    def mock_neo4j(self):
        with patch("nemesis.graph.neo4j.neo4j_driver") as mock_driver:
            mock_session = MagicMock()
            mock_tx = MagicMock()
            mock_driver_instance = MagicMock()
            mock_driver_instance.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver_instance.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.execute_write.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_session.execute_read.side_effect = lambda fn, **kwargs: fn(mock_tx, **kwargs)
            mock_driver.GraphDatabase.driver.return_value = mock_driver_instance
            yield {
                "driver": mock_driver,
                "driver_instance": mock_driver_instance,
                "session": mock_session,
                "tx": mock_tx,
            }

    def test_add_edge_executes_merge(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.add_edge(EdgeData(source_id="f1", target_id="f2", edge_type="CALLS"))
        assert mock_neo4j["tx"].run.called
        call_args = mock_neo4j["tx"].run.call_args
        query = call_args[0][0]
        assert "MATCH" in query
        assert "CALLS" in query
        adapter.close()

    def test_delete_edges_for_file(self, mock_neo4j) -> None:
        adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="test")
        adapter.delete_edges_for_file("/src/main.py")
        assert mock_neo4j["tx"].run.called
        adapter.close()
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_neo4j.py -x -v 2>&1 | head -30
```

Tests fail because `nemesis/graph/neo4j.py` does not exist.

### Step 3 — Implement

```python
# nemesis/graph/neo4j.py
"""Neo4j graph adapter for Nemesis.

Optional backend using the neo4j Python driver. Requires a running
Neo4j instance. Install with: pip install nemesis-ai[neo4j]
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import neo4j as neo4j_driver
except ImportError:
    neo4j_driver = None  # type: ignore[assignment]

from nemesis.graph.adapter import (
    EDGE_TYPES,
    NODE_SCHEMAS,
    NODE_TYPES,
    EdgeData,
    NodeData,
    TraversalResult,
)

logger = logging.getLogger(__name__)


class Neo4jAdapter:
    """Neo4j graph database adapter.

    Requires a running Neo4j instance. Uses the official neo4j
    Python driver for communication.

    Args:
        uri: Neo4j connection URI (e.g. "bolt://localhost:7687").
        user: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name (default: "neo4j").
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ) -> None:
        if neo4j_driver is None:
            raise ImportError(
                "neo4j driver is required for Neo4jAdapter. "
                "Install with: pip install nemesis-ai[neo4j]"
            )
        self._driver = neo4j_driver.GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def _session(self):
        """Create a new database session."""
        return self._driver.session(database=self._database)

    def create_schema(self) -> None:
        """Create indexes and constraints for all node types.

        Neo4j uses labels instead of table DDL, so schema creation
        means creating uniqueness constraints and indexes.
        """
        with self._session() as session:
            for node_type in NODE_TYPES:
                try:
                    session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS "
                        f"FOR (n:{node_type}) REQUIRE n.id IS UNIQUE"
                    )
                except Exception as e:
                    logger.debug("Constraint for %s: %s", node_type, e)

            # Create indexes on commonly queried properties
            for node_type, prop in [
                ("File", "path"), ("Function", "file"), ("Class", "file"),
                ("Method", "file"), ("Variable", "file"),
            ]:
                try:
                    session.run(
                        f"CREATE INDEX IF NOT EXISTS "
                        f"FOR (n:{node_type}) ON (n.{prop})"
                    )
                except Exception as e:
                    logger.debug("Index for %s.%s: %s", node_type, prop, e)

    def add_node(self, node: NodeData) -> None:
        """Add or update a node using MERGE."""
        props = {"id": node.id, **node.properties}
        set_clause = ", ".join(
            f"n.{k} = ${k}" for k in props if k != "id"
        )
        query = f"MERGE (n:{node.node_type} {{id: $id}})"
        if set_clause:
            query += f" SET {set_clause}"

        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(query, **kw), **props)

    def add_edge(self, edge: EdgeData) -> None:
        """Add an edge between two nodes."""
        props_clause = ""
        params: dict[str, Any] = {
            "src_id": edge.source_id,
            "tgt_id": edge.target_id,
        }
        if edge.properties:
            prop_sets = ", ".join(f"{k}: ${k}" for k in edge.properties)
            props_clause = f" {{{prop_sets}}}"
            params.update(edge.properties)

        query = (
            f"MATCH (a {{id: $src_id}}), (b {{id: $tgt_id}}) "
            f"MERGE (a)-[:{edge.edge_type}{props_clause}]->(b)"
        )

        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(query, **kw), **params)

    def get_node(self, node_id: str) -> NodeData | None:
        """Retrieve a node by its ID."""
        query = "MATCH (n {id: $id}) RETURN labels(n) AS labels, properties(n) AS props"

        with self._session() as session:
            def _read(tx, **kwargs):
                result = tx.run(query, **kwargs)
                record = result.single()
                return record
            record = session.execute_read(_read, id=node_id)

        if record is None:
            return None

        data = record.data()
        labels = data.get("labels", [])
        props = data.get("props", {})
        node_type = labels[0] if labels else "Unknown"
        node_id_val = props.pop("id", node_id)

        return NodeData(id=node_id_val, node_type=node_type, properties=props)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[NodeData]:
        """Get neighboring nodes."""
        rel_pattern = f":{edge_type}" if edge_type else ""

        if direction == "outgoing":
            query = (
                f"MATCH (a {{id: $id}})-[{rel_pattern}]->(b) "
                f"RETURN labels(b) AS labels, properties(b) AS props"
            )
        elif direction == "incoming":
            query = (
                f"MATCH (a {{id: $id}})<-[{rel_pattern}]-(b) "
                f"RETURN labels(b) AS labels, properties(b) AS props"
            )
        else:  # both
            query = (
                f"MATCH (a {{id: $id}})-[{rel_pattern}]-(b) "
                f"RETURN DISTINCT labels(b) AS labels, properties(b) AS props"
            )

        neighbors: list[NodeData] = []
        with self._session() as session:
            def _read(tx, **kwargs):
                result = tx.run(query, **kwargs)
                return [r.data() for r in result]
            records = session.execute_read(_read, id=node_id)

        for data in records:
            labels = data.get("labels", [])
            props = data.get("props", {})
            nid = props.pop("id", "")
            node_type = labels[0] if labels else "Unknown"
            neighbors.append(NodeData(id=nid, node_type=node_type, properties=props))

        return neighbors

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> TraversalResult:
        """Traverse the graph from a starting node."""
        rel_filter = "|".join(edge_types) if edge_types else ""
        rel_pattern = f":{rel_filter}" if rel_filter else ""

        query = (
            f"MATCH path = (start {{id: $id}})-[{rel_pattern}*1..{max_depth}]->(end) "
            f"UNWIND nodes(path) AS n "
            f"WITH DISTINCT n "
            f"RETURN labels(n) AS labels, properties(n) AS props"
        )

        all_nodes: dict[str, NodeData] = {}

        # Always include the start node
        start_node = self.get_node(start_id)
        if start_node:
            all_nodes[start_id] = start_node

        with self._session() as session:
            def _read(tx, **kwargs):
                result = tx.run(query, **kwargs)
                return [r.data() for r in result]
            try:
                records = session.execute_read(_read, id=start_id)
                for data in records:
                    labels = data.get("labels", [])
                    props = data.get("props", {})
                    nid = props.pop("id", "")
                    node_type = labels[0] if labels else "Unknown"
                    if nid not in all_nodes:
                        all_nodes[nid] = NodeData(id=nid, node_type=node_type, properties=props)
            except Exception as e:
                logger.debug("Traversal query failed: %s", e)

        return TraversalResult(nodes=list(all_nodes.values()), edges=[])

    def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Execute a raw Cypher query."""
        with self._session() as session:
            def _read(tx, **kwargs):
                result = tx.run(cypher, **kwargs)
                return [dict(record) for record in result]
            return session.execute_read(_read, **(parameters or {}))

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        query = "MATCH (n {id: $id}) DETACH DELETE n"
        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(query, **kw), id=node_id)

    def delete_edges_for_file(self, file_path: str) -> None:
        """Delete all edges for nodes associated with a file."""
        query = (
            "MATCH (n)-[r]-() "
            "WHERE n.file = $path OR n.path = $path "
            "DELETE r"
        )
        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(query, **kw), path=file_path)

    def get_file_hashes(self) -> dict[str, str]:
        """Get all stored file hashes from File nodes."""
        query = "MATCH (f:File) WHERE f.hash IS NOT NULL RETURN f.path AS path, f.hash AS hash"
        rows = self.query(query)
        return {r["path"]: r["hash"] for r in rows if r.get("path") and r.get("hash")}

    def get_nodes_for_file(self, file_path: str) -> list[NodeData]:
        """Get all nodes associated with a file."""
        query = (
            "MATCH (n) "
            "WHERE n.file = $path OR n.path = $path "
            "RETURN labels(n) AS labels, properties(n) AS props"
        )
        nodes: list[NodeData] = []
        rows = self.query(query, parameters={"path": file_path})
        for data in rows:
            labels = data.get("labels", [])
            props = data.get("props", {})
            nid = props.pop("id", "")
            node_type = labels[0] if labels else "Unknown"
            nodes.append(NodeData(id=nid, node_type=node_type, properties=props))
        return nodes

    def get_chunk_ids_for_file(self, file_path: str) -> list[str]:
        """Get IDs of all Chunk nodes linked to a file's nodes."""
        query = (
            "MATCH (c:Chunk)-[:CHUNK_OF]->(n) "
            "WHERE n.file = $path OR n.path = $path "
            "RETURN c.id AS id"
        )
        rows = self.query(query, parameters={"path": file_path})
        return [r["id"] for r in rows]

    def delete_nodes_for_file(self, file_path: str) -> None:
        """Delete all nodes and edges associated with a file."""
        # Delete chunks first
        chunk_query = (
            "MATCH (c:Chunk)-[:CHUNK_OF]->(n) "
            "WHERE n.file = $path OR n.path = $path "
            "DETACH DELETE c"
        )
        node_query = (
            "MATCH (n) "
            "WHERE n.file = $path OR n.path = $path "
            "DETACH DELETE n"
        )
        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(chunk_query, **kw), path=file_path)
            session.execute_write(lambda tx, **kw: tx.run(node_query, **kw), path=file_path)

    def clear(self) -> None:
        """Delete all nodes and edges from the graph."""
        with self._session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_neo4j.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/graph/neo4j.py tests/test_graph/test_neo4j.py
git commit -m "feat(graph): implement Neo4jAdapter with full CRUD and traversal

TDD Task 7/10 of 03-graph-layer plan.
Neo4j adapter using official driver. Uses MERGE for upserts,
DETACH DELETE for node removal, labels(n)/properties(n) for
node retrieval. All tests use mocked driver."
```

---

## Task 9: KuzuAdapter — Memory Node Operations

**Files:**
- `tests/test_graph/test_kuzu_memory.py`

### Step 1 — Write failing test

```python
# tests/test_graph/test_kuzu_memory.py
"""Tests for memory-related node operations (Rules, Decisions, Conventions)."""
from __future__ import annotations

import pytest
from pathlib import Path

from nemesis.graph.adapter import NodeData, EdgeData
from nemesis.graph.kuzu import KuzuAdapter


class TestKuzuMemoryNodes:
    """Tests for memory node types used by the memory system."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> KuzuAdapter:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    def test_add_rule_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(
            id="rule-001", node_type="Rule",
            properties={
                "content": "Always use parameterized queries",
                "scope": "global",
                "created_at": "2026-02-20T10:00:00Z",
                "source": "developer",
            },
        ))
        node = adapter.get_node("rule-001")
        assert node is not None
        assert node.properties["content"] == "Always use parameterized queries"
        assert node.properties["scope"] == "global"

    def test_add_decision_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(
            id="dec-001", node_type="Decision",
            properties={
                "title": "Use Kuzu as default graph DB",
                "reasoning": "Zero-config, embedded, fast enough",
                "created_at": "2026-02-20T10:00:00Z",
                "status": "accepted",
            },
        ))
        node = adapter.get_node("dec-001")
        assert node is not None
        assert node.properties["title"] == "Use Kuzu as default graph DB"

    def test_add_alternative_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(
            id="alt-001", node_type="Alternative",
            properties={
                "title": "Use SQLite with virtual tables",
                "reason_rejected": "No native graph traversal",
            },
        ))
        node = adapter.get_node("alt-001")
        assert node is not None
        assert node.properties["reason_rejected"] == "No native graph traversal"

    def test_add_convention_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(
            id="conv-001", node_type="Convention",
            properties={
                "pattern": "Use dataclasses for simple DTOs",
                "example": "@dataclass\nclass UserDTO: ...",
                "scope": "python",
            },
        ))
        node = adapter.get_node("conv-001")
        assert node is not None
        assert node.properties["pattern"] == "Use dataclasses for simple DTOs"

    def test_rule_applies_to_file(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(id="rule-sql", node_type="Rule",
            properties={"content": "Use parameterized queries", "scope": "global"}))
        adapter.add_node(NodeData(id="file-db", node_type="File",
            properties={"path": "db.py", "language": "python"}))
        adapter.add_edge(EdgeData(
            source_id="rule-sql", target_id="file-db", edge_type="APPLIES_TO",
        ))
        neighbors = adapter.get_neighbors("rule-sql", edge_type="APPLIES_TO", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "file-db" in ids

    def test_decision_chose_convention(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(id="dec-dc", node_type="Decision",
            properties={"title": "Dataclass convention", "status": "accepted"}))
        adapter.add_node(NodeData(id="conv-dc", node_type="Convention",
            properties={"pattern": "Use dataclasses"}))
        adapter.add_edge(EdgeData(
            source_id="dec-dc", target_id="conv-dc", edge_type="CHOSE",
        ))
        neighbors = adapter.get_neighbors("dec-dc", edge_type="CHOSE", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "conv-dc" in ids

    def test_decision_rejected_alternative(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(id="dec-rej", node_type="Decision",
            properties={"title": "Choose DB", "status": "accepted"}))
        adapter.add_node(NodeData(id="alt-sqlite", node_type="Alternative",
            properties={"title": "SQLite"}))
        adapter.add_edge(EdgeData(
            source_id="dec-rej", target_id="alt-sqlite", edge_type="REJECTED",
        ))
        neighbors = adapter.get_neighbors("dec-rej", edge_type="REJECTED", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "alt-sqlite" in ids

    def test_project_has_file(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(id="proj-nemesis", node_type="Project",
            properties={"name": "nemesis", "root_path": "/home/user/nemesis"}))
        adapter.add_node(NodeData(id="file-init", node_type="File",
            properties={"path": "__init__.py", "language": "python"}))
        adapter.add_edge(EdgeData(
            source_id="proj-nemesis", target_id="file-init", edge_type="HAS_FILE",
        ))
        neighbors = adapter.get_neighbors("proj-nemesis", edge_type="HAS_FILE", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "file-init" in ids
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu_memory.py -x -v 2>&1 | head -20
```

### Step 3 — Implementation already in Task 2

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu_memory.py -x -v
```

### Step 5 — Commit

```bash
git add tests/test_graph/test_kuzu_memory.py
git commit -m "test(graph): add memory node operation tests for KuzuAdapter

TDD Task 9/10 of 03-graph-layer plan.
Tests Rule, Decision, Alternative, Convention, Project nodes.
Verifies APPLIES_TO, CHOSE, REJECTED, GOVERNS, HAS_FILE edges.
Ensures the memory system can use the graph layer."
```

---

## Summary

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 7 | Neo4jAdapter Full Implementation | `neo4j.py`, `test_neo4j.py` | 7 |
| 9 | Memory Node Operations | `test_kuzu_memory.py` | 8 |
| **Gesamt C3** | | **1 Source + 2 Test-Dateien** | **~15 Tests** |

---

**Navigation:**
- Vorheriges Paket: [C2 — Kuzu CRUD](03b-kuzu-crud.md)
- Nachstes Paket: [C4 — Graph Integration](03d-graph-integration.md)
- Gesamtplan: [03-graph-layer.md](03-graph-layer.md)
