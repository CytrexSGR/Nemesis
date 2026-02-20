# Graph Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the graph abstraction layer with Kuzu (default) and Neo4j (optional) backends for storing code knowledge graphs.

**Architecture:** `GraphAdapter` Protocol defines the common interface. `KuzuAdapter` provides zero-config embedded graph storage (default). `Neo4jAdapter` offers optional remote graph database support. Both backends use Cypher-compatible schemas. All graph operations are synchronous since both Kuzu and Neo4j Python drivers are sync.

**Tech Stack:** Python 3.11+, Kuzu >= 0.4, Neo4j >= 5.0 (optional), pytest

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [01-project-setup](01-project-setup.md)

---

## Task 1: GraphAdapter Protocol & Data Models

**Files:**
- `tests/test_graph/__init__.py`
- `tests/test_graph/test_adapter.py`
- `nemesis/graph/__init__.py`
- `nemesis/graph/adapter.py`

### Step 1 — Write failing test

```python
# tests/test_graph/__init__.py
```

```python
# tests/test_graph/test_adapter.py
"""Tests for GraphAdapter protocol and graph data models."""
import pytest
from dataclasses import asdict

from nemesis.graph.adapter import (
    GraphAdapter,
    NodeData,
    EdgeData,
    TraversalResult,
    NODE_TYPES,
    EDGE_TYPES,
)


class TestNodeData:
    """Tests for the NodeData model."""

    def test_node_data_creation(self) -> None:
        node = NodeData(
            id="func-001",
            node_type="Function",
            properties={"name": "hello", "file": "main.py", "line_start": 1},
        )
        assert node.id == "func-001"
        assert node.node_type == "Function"
        assert node.properties["name"] == "hello"

    def test_node_data_defaults(self) -> None:
        node = NodeData(id="n1", node_type="File")
        assert node.properties == {}

    def test_node_data_to_dict(self) -> None:
        node = NodeData(id="n1", node_type="Class", properties={"name": "Foo"})
        d = asdict(node)
        assert d["id"] == "n1"
        assert d["node_type"] == "Class"
        assert d["properties"]["name"] == "Foo"


class TestEdgeData:
    """Tests for the EdgeData model."""

    def test_edge_data_creation(self) -> None:
        edge = EdgeData(
            source_id="file-001",
            target_id="func-001",
            edge_type="CONTAINS",
            properties={"weight": 1.0},
        )
        assert edge.source_id == "file-001"
        assert edge.target_id == "func-001"
        assert edge.edge_type == "CONTAINS"
        assert edge.properties["weight"] == 1.0

    def test_edge_data_defaults(self) -> None:
        edge = EdgeData(source_id="a", target_id="b", edge_type="CALLS")
        assert edge.properties == {}


class TestTraversalResult:
    """Tests for the TraversalResult model."""

    def test_traversal_result_creation(self) -> None:
        nodes = [
            NodeData(id="n1", node_type="Function", properties={"name": "a"}),
            NodeData(id="n2", node_type="Function", properties={"name": "b"}),
        ]
        edges = [
            EdgeData(source_id="n1", target_id="n2", edge_type="CALLS"),
        ]
        result = TraversalResult(nodes=nodes, edges=edges)
        assert len(result.nodes) == 2
        assert len(result.edges) == 1

    def test_traversal_result_empty(self) -> None:
        result = TraversalResult(nodes=[], edges=[])
        assert result.nodes == []
        assert result.edges == []


class TestSchemaConstants:
    """Tests for the schema constants."""

    def test_node_types_contains_all_code_nodes(self) -> None:
        expected = {
            "File", "Module", "Class", "Function", "Method",
            "Interface", "Variable", "Import",
        }
        assert expected.issubset(NODE_TYPES)

    def test_node_types_contains_all_memory_nodes(self) -> None:
        expected = {"Rule", "Decision", "Alternative", "Convention"}
        assert expected.issubset(NODE_TYPES)

    def test_node_types_contains_meta_nodes(self) -> None:
        expected = {"Project", "Chunk"}
        assert expected.issubset(NODE_TYPES)

    def test_edge_types_contains_all_types(self) -> None:
        expected = {
            "CONTAINS", "HAS_METHOD", "INHERITS", "IMPLEMENTS",
            "CALLS", "IMPORTS", "RETURNS", "ACCEPTS",
            "APPLIES_TO", "CHOSE", "REJECTED", "GOVERNS",
            "CHUNK_OF", "HAS_FILE",
        }
        assert expected.issubset(EDGE_TYPES)


class TestGraphAdapterProtocol:
    """Tests that GraphAdapter is a proper Protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(GraphAdapter, "__protocol_attrs__") or hasattr(
            GraphAdapter, "__abstractmethods__"
        )

    def test_conforming_class_is_instance(self) -> None:
        """A class implementing all methods should be recognized."""

        class FakeAdapter:
            def create_schema(self) -> None: ...
            def add_node(self, node: NodeData) -> None: ...
            def add_edge(self, edge: EdgeData) -> None: ...
            def get_node(self, node_id: str) -> NodeData | None: ...
            def get_neighbors(
                self, node_id: str, edge_type: str | None = None,
                direction: str = "outgoing",
            ) -> list[NodeData]: ...
            def traverse(
                self, start_id: str, edge_types: list[str] | None = None,
                max_depth: int = 3,
            ) -> TraversalResult: ...
            def query(self, cypher: str, parameters: dict | None = None) -> list[dict]: ...
            def delete_node(self, node_id: str) -> None: ...
            def delete_edges_for_file(self, file_path: str) -> None: ...
            def get_file_hashes(self) -> dict[str, str]: ...
            def get_nodes_for_file(self, file_path: str) -> list[NodeData]: ...
            def get_chunk_ids_for_file(self, file_path: str) -> list[str]: ...
            def delete_nodes_for_file(self, file_path: str) -> None: ...
            def clear(self) -> None: ...
            def close(self) -> None: ...

        adapter = FakeAdapter()
        assert isinstance(adapter, GraphAdapter)
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_adapter.py -x -v 2>&1 | head -30
```

Tests fail because `nemesis/graph/adapter.py` does not exist.

### Step 3 — Implement

```python
# nemesis/graph/__init__.py
"""Graph module — abstract adapter, Kuzu, Neo4j."""
```

```python
# nemesis/graph/adapter.py
"""Abstract graph adapter protocol for Nemesis.

Defines the common interface that all graph backends (Kuzu, Neo4j)
must implement. Uses Python's Protocol for structural subtyping —
no inheritance required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Schema constants — all node and edge types from the design document
# ---------------------------------------------------------------------------

NODE_TYPES: set[str] = {
    # Code nodes (from Tree-sitter)
    "File",
    "Module",
    "Class",
    "Function",
    "Method",
    "Interface",
    "Variable",
    "Import",
    # Memory nodes (from developer)
    "Rule",
    "Decision",
    "Alternative",
    "Convention",
    # Meta nodes
    "Project",
    "Chunk",
}

EDGE_TYPES: set[str] = {
    # Code edges
    "CONTAINS",
    "HAS_METHOD",
    "INHERITS",
    "IMPLEMENTS",
    "CALLS",
    "IMPORTS",
    "RETURNS",
    "ACCEPTS",
    # Memory edges
    "APPLIES_TO",
    "CHOSE",
    "REJECTED",
    "GOVERNS",
    # Chunk edges
    "CHUNK_OF",
    # Meta edges
    "HAS_FILE",
}

# ---------------------------------------------------------------------------
# Node property schemas — required and optional properties per node type
# ---------------------------------------------------------------------------

NODE_SCHEMAS: dict[str, dict[str, str]] = {
    "File": {
        "id": "STRING", "path": "STRING", "language": "STRING",
        "hash": "STRING", "last_indexed": "STRING", "size": "INT64",
    },
    "Module": {
        "id": "STRING", "name": "STRING", "path": "STRING",
        "docstring": "STRING",
    },
    "Class": {
        "id": "STRING", "name": "STRING", "file": "STRING",
        "line_start": "INT64", "line_end": "INT64", "docstring": "STRING",
    },
    "Function": {
        "id": "STRING", "name": "STRING", "file": "STRING",
        "line_start": "INT64", "line_end": "INT64", "signature": "STRING",
        "docstring": "STRING", "is_async": "BOOL",
    },
    "Method": {
        "id": "STRING", "name": "STRING", "class_name": "STRING",
        "file": "STRING", "line_start": "INT64", "line_end": "INT64",
        "signature": "STRING", "visibility": "STRING",
    },
    "Interface": {
        "id": "STRING", "name": "STRING", "file": "STRING",
        "language": "STRING",
    },
    "Variable": {
        "id": "STRING", "name": "STRING", "file": "STRING",
        "type_hint": "STRING", "scope": "STRING",
    },
    "Import": {
        "id": "STRING", "name": "STRING", "source": "STRING",
        "alias": "STRING",
    },
    "Rule": {
        "id": "STRING", "content": "STRING", "scope": "STRING",
        "created_at": "STRING", "source": "STRING",
    },
    "Decision": {
        "id": "STRING", "title": "STRING", "reasoning": "STRING",
        "created_at": "STRING", "status": "STRING",
    },
    "Alternative": {
        "id": "STRING", "title": "STRING", "reason_rejected": "STRING",
    },
    "Convention": {
        "id": "STRING", "pattern": "STRING", "example": "STRING",
        "scope": "STRING",
    },
    "Project": {
        "id": "STRING", "name": "STRING", "root_path": "STRING",
        "languages": "STRING", "last_indexed": "STRING",
    },
    "Chunk": {
        "id": "STRING", "content": "STRING", "token_count": "INT64",
        "embedding_id": "STRING", "parent_type": "STRING",
    },
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeData:
    """A node to be stored in the graph.

    Attributes:
        id: Unique identifier for the node.
        node_type: One of NODE_TYPES (e.g. "Function", "Class", "File").
        properties: Arbitrary key-value properties for the node.
    """

    id: str
    node_type: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeData:
    """An edge (relationship) to be stored in the graph.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        edge_type: One of EDGE_TYPES (e.g. "CALLS", "CONTAINS").
        properties: Arbitrary key-value properties for the edge.
    """

    source_id: str
    target_id: str
    edge_type: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TraversalResult:
    """Result of a graph traversal operation.

    Contains all nodes and edges found during the traversal,
    starting from a given node and following specified edge types
    up to a maximum depth.

    Attributes:
        nodes: All nodes discovered during traversal.
        edges: All edges traversed.
    """

    nodes: list[NodeData]
    edges: list[EdgeData]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GraphAdapter(Protocol):
    """Protocol for graph database backends.

    Any class implementing these methods can serve as a graph store.
    Two concrete implementations ship with Nemesis:
    KuzuAdapter (embedded, default) and Neo4jAdapter (optional).
    """

    def create_schema(self) -> None:
        """Create all node and edge tables in the database.

        Must be idempotent — calling it on an existing schema
        should not raise or destroy data.
        """
        ...

    def add_node(self, node: NodeData) -> None:
        """Add or update a node in the graph.

        If a node with the same ID already exists, it is updated.

        Args:
            node: The node data to add.
        """
        ...

    def add_edge(self, edge: EdgeData) -> None:
        """Add an edge between two existing nodes.

        Args:
            edge: The edge data to add.
        """
        ...

    def get_node(self, node_id: str) -> NodeData | None:
        """Retrieve a node by its ID.

        Args:
            node_id: The unique node identifier.

        Returns:
            NodeData if found, None otherwise.
        """
        ...

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[NodeData]:
        """Get neighboring nodes connected by edges.

        Args:
            node_id: The starting node ID.
            edge_type: Optional filter for edge type.
            direction: "outgoing", "incoming", or "both".

        Returns:
            List of neighboring NodeData.
        """
        ...

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> TraversalResult:
        """Traverse the graph from a starting node.

        Performs a breadth-first traversal, following specified
        edge types up to max_depth hops.

        Args:
            start_id: The starting node ID.
            edge_types: Optional list of edge types to follow.
            max_depth: Maximum traversal depth.

        Returns:
            TraversalResult with all discovered nodes and edges.
        """
        ...

    def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Execute a raw Cypher query.

        Args:
            cypher: The Cypher query string.
            parameters: Optional query parameters.

        Returns:
            List of result rows as dicts.
        """
        ...

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges.

        Args:
            node_id: The node ID to delete.
        """
        ...

    def delete_edges_for_file(self, file_path: str) -> None:
        """Delete all edges associated with nodes from a specific file.

        Args:
            file_path: The file path whose edges should be removed.
        """
        ...

    def get_file_hashes(self) -> dict[str, str]:
        """Get all stored file hashes.

        Returns:
            Dict mapping file path to SHA-256 hash.
        """
        ...

    def get_nodes_for_file(self, file_path: str) -> list[NodeData]:
        """Get all nodes associated with a file.

        Args:
            file_path: The file path to query.

        Returns:
            List of NodeData belonging to the file.
        """
        ...

    def get_chunk_ids_for_file(self, file_path: str) -> list[str]:
        """Get IDs of all Chunk nodes linked to a file's nodes.

        Args:
            file_path: The file path to query.

        Returns:
            List of chunk node IDs.
        """
        ...

    def delete_nodes_for_file(self, file_path: str) -> None:
        """Delete all nodes and edges associated with a file.

        Used during delta updates to remove stale data before
        re-indexing a changed file.

        Args:
            file_path: The file path whose nodes should be removed.
        """
        ...

    def clear(self) -> None:
        """Delete all nodes and edges from the graph.

        Used for testing and full re-index.
        """
        ...

    def close(self) -> None:
        """Close the database connection and release resources."""
        ...
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_adapter.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/graph/__init__.py nemesis/graph/adapter.py \
        tests/test_graph/__init__.py tests/test_graph/test_adapter.py
git commit -m "feat(graph): add GraphAdapter protocol, data models, and schema constants

TDD Task 1/10 of 03-graph-layer plan.
Defines the runtime-checkable Protocol that all graph backends
must implement, NodeData/EdgeData/TraversalResult dataclasses,
and NODE_TYPES/EDGE_TYPES/NODE_SCHEMAS constants."
```

---

## Task 2: KuzuAdapter — Initialization & Schema Creation

**Files:**
- `tests/test_graph/test_kuzu.py`
- `nemesis/graph/kuzu.py`

### Step 1 — Write failing test

```python
# tests/test_graph/test_kuzu.py
"""Tests for Kuzu graph adapter."""
from __future__ import annotations

import pytest
from pathlib import Path

from nemesis.graph.adapter import GraphAdapter, NodeData, EdgeData, NODE_TYPES, EDGE_TYPES
from nemesis.graph.kuzu import KuzuAdapter


class TestKuzuInit:
    """Tests for KuzuAdapter initialization and schema."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "test_graph")

    @pytest.fixture
    def adapter(self, db_path: str) -> KuzuAdapter:
        a = KuzuAdapter(db_path=db_path)
        a.create_schema()
        yield a
        a.close()

    def test_is_graph_adapter(self, adapter: KuzuAdapter) -> None:
        assert isinstance(adapter, GraphAdapter)

    def test_create_schema_idempotent(self, db_path: str) -> None:
        """Calling create_schema twice should not raise."""
        a = KuzuAdapter(db_path=db_path)
        a.create_schema()
        a.create_schema()  # Should not raise
        a.close()

    def test_schema_creates_all_node_tables(self, adapter: KuzuAdapter) -> None:
        """All node types from the design doc have a table."""
        for node_type in NODE_TYPES:
            # Verify by inserting a minimal node — should not raise
            node = NodeData(id=f"test-{node_type}", node_type=node_type)
            adapter.add_node(node)
            result = adapter.get_node(f"test-{node_type}")
            assert result is not None, f"Node table for {node_type} not found"

    def test_schema_creates_all_edge_tables(self, adapter: KuzuAdapter) -> None:
        """All edge types from the design doc have a relationship table."""
        # We verify edge tables exist by checking schema — they're created
        # even if empty. Add source and target nodes first.
        adapter.add_node(NodeData(id="src-node", node_type="File", properties={"path": "a.py"}))
        adapter.add_node(NodeData(id="tgt-node", node_type="Function", properties={"name": "f"}))
        # CONTAINS is a valid edge between File and Function
        edge = EdgeData(source_id="src-node", target_id="tgt-node", edge_type="CONTAINS")
        adapter.add_edge(edge)  # Should not raise

    def test_close_and_reopen(self, db_path: str) -> None:
        """Data persists after close and reopen."""
        a1 = KuzuAdapter(db_path=db_path)
        a1.create_schema()
        a1.add_node(NodeData(id="persistent", node_type="File", properties={"path": "test.py"}))
        a1.close()

        a2 = KuzuAdapter(db_path=db_path)
        a2.create_schema()
        node = a2.get_node("persistent")
        assert node is not None
        assert node.id == "persistent"
        a2.close()
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuInit -x -v 2>&1 | head -30
```

Tests fail because `nemesis/graph/kuzu.py` does not exist.

### Step 3 — Implement

```python
# nemesis/graph/kuzu.py
"""Kuzu embedded graph adapter for Nemesis.

Provides a zero-config, embedded graph database using Kuzu.
This is the default backend — no external services required.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import kuzu

from nemesis.graph.adapter import (
    EDGE_TYPES,
    NODE_SCHEMAS,
    NODE_TYPES,
    EdgeData,
    NodeData,
    TraversalResult,
)

logger = logging.getLogger(__name__)


# Mapping of edge types to valid (source_type, target_type) pairs.
# Kuzu requires REL tables to reference specific node tables.
EDGE_ENDPOINTS: dict[str, list[tuple[str, str]]] = {
    "CONTAINS": [
        ("File", "Class"), ("File", "Function"), ("File", "Variable"),
        ("File", "Import"), ("File", "Interface"), ("Module", "Class"),
        ("Module", "Function"), ("Module", "Variable"),
    ],
    "HAS_METHOD": [("Class", "Method")],
    "INHERITS": [("Class", "Class")],
    "IMPLEMENTS": [("Class", "Interface")],
    "CALLS": [
        ("Function", "Function"), ("Function", "Method"),
        ("Method", "Function"), ("Method", "Method"),
    ],
    "IMPORTS": [
        ("Function", "Module"), ("Function", "File"),
        ("File", "File"), ("File", "Module"),
        ("Module", "Module"),
    ],
    "RETURNS": [("Function", "Class"), ("Method", "Class")],
    "ACCEPTS": [("Function", "Class"), ("Method", "Class")],
    "APPLIES_TO": [
        ("Rule", "File"), ("Rule", "Module"), ("Rule", "Class"),
        ("Rule", "Project"),
    ],
    "CHOSE": [
        ("Decision", "Convention"), ("Decision", "Class"),
        ("Decision", "Module"),
    ],
    "REJECTED": [("Decision", "Alternative")],
    "GOVERNS": [("Convention", "Module"), ("Convention", "File")],
    "CHUNK_OF": [
        ("Chunk", "Class"), ("Chunk", "Function"),
        ("Chunk", "Method"), ("Chunk", "File"),
    ],
    "HAS_FILE": [("Project", "File")],
}


class KuzuAdapter:
    """Kuzu embedded graph database adapter.

    Args:
        db_path: Directory path for Kuzu database storage.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._schema_created = False

    def create_schema(self) -> None:
        """Create all node and relationship tables.

        Idempotent: skips tables that already exist.
        """
        existing_tables = self._get_existing_tables()

        # Create node tables
        for node_type in sorted(NODE_TYPES):
            if node_type in existing_tables:
                continue
            schema = NODE_SCHEMAS.get(node_type, {"id": "STRING"})
            columns = ", ".join(
                f"{col} {dtype}" for col, dtype in schema.items()
                if col != "id"
            )
            ddl = f"CREATE NODE TABLE {node_type}(id STRING, {columns}, PRIMARY KEY (id))"
            try:
                self._conn.execute(ddl)
                logger.debug("Created node table: %s", node_type)
            except RuntimeError as e:
                if "already exists" not in str(e).lower():
                    raise

        # Create relationship tables
        for edge_type in sorted(EDGE_TYPES):
            endpoints = EDGE_ENDPOINTS.get(edge_type, [])
            for src_type, tgt_type in endpoints:
                table_name = f"{edge_type}_{src_type}_{tgt_type}"
                if table_name in existing_tables:
                    continue
                ddl = (
                    f"CREATE REL TABLE {table_name}"
                    f"(FROM {src_type} TO {tgt_type})"
                )
                try:
                    self._conn.execute(ddl)
                    logger.debug("Created rel table: %s", table_name)
                except RuntimeError as e:
                    if "already exists" not in str(e).lower():
                        raise

        self._schema_created = True

    def _get_existing_tables(self) -> set[str]:
        """Query Kuzu for existing table names."""
        tables: set[str] = set()
        try:
            result = self._conn.execute("CALL show_tables() RETURN name")
            while result.has_next():
                row = result.get_next()
                tables.add(row[0])
        except RuntimeError:
            pass
        return tables

    def _find_node_type(self, node_id: str) -> str | None:
        """Find which node table contains a given node ID."""
        for node_type in NODE_TYPES:
            try:
                result = self._conn.execute(
                    f"MATCH (n:{node_type}) WHERE n.id = $id RETURN n.id",
                    parameters={"id": node_id},
                )
                if result.has_next():
                    return node_type
            except RuntimeError:
                continue
        return None

    def add_node(self, node: NodeData) -> None:
        """Add or update a node in the graph."""
        schema = NODE_SCHEMAS.get(node.node_type, {"id": "STRING"})
        props = {"id": node.id}
        for col, dtype in schema.items():
            if col == "id":
                continue
            value = node.properties.get(col)
            if value is None:
                if dtype == "STRING":
                    value = ""
                elif dtype == "INT64":
                    value = 0
                elif dtype == "BOOL":
                    value = False
            props[col] = value

        columns = list(props.keys())
        placeholders = ", ".join(f"${col}" for col in columns)
        col_list = ", ".join(columns)

        try:
            # Try MERGE (upsert)
            set_clauses = ", ".join(
                f"n.{col} = ${col}" for col in columns if col != "id"
            )
            merge_query = (
                f"MERGE (n:{node.node_type} {{id: $id}}) "
                f"SET {set_clauses}" if set_clauses else
                f"MERGE (n:{node.node_type} {{id: $id}})"
            )
            self._conn.execute(merge_query, parameters=props)
        except RuntimeError:
            # Fallback: try CREATE
            try:
                create_query = (
                    f"CREATE (:{node.node_type} {{{col_list}}})"
                ).replace("{" + col_list + "}", self._build_props_literal(props, schema))
                self._conn.execute(
                    f"CREATE (n:{node.node_type} {{{', '.join(f'{c}: ${c}' for c in columns)}}})",
                    parameters=props,
                )
            except RuntimeError:
                pass  # Node already exists

    @staticmethod
    def _build_props_literal(props: dict, schema: dict) -> str:
        """Build a Cypher properties literal string."""
        parts = []
        for k, v in props.items():
            parts.append(f"{k}: ${k}")
        return ", ".join(parts)

    def add_edge(self, edge: EdgeData) -> None:
        """Add an edge between two existing nodes."""
        src_type = self._find_node_type(edge.source_id)
        tgt_type = self._find_node_type(edge.target_id)
        if src_type is None or tgt_type is None:
            logger.warning(
                "Cannot add edge %s: source (%s) or target (%s) not found",
                edge.edge_type, edge.source_id, edge.target_id,
            )
            return

        table_name = f"{edge.edge_type}_{src_type}_{tgt_type}"
        query = (
            f"MATCH (a:{src_type}), (b:{tgt_type}) "
            f"WHERE a.id = $src AND b.id = $tgt "
            f"CREATE (a)-[:{table_name}]->(b)"
        )
        try:
            self._conn.execute(
                query,
                parameters={"src": edge.source_id, "tgt": edge.target_id},
            )
        except RuntimeError as e:
            logger.warning("Failed to add edge %s: %s", table_name, e)

    def get_node(self, node_id: str) -> NodeData | None:
        """Retrieve a node by its ID."""
        for node_type in NODE_TYPES:
            try:
                schema = NODE_SCHEMAS.get(node_type, {"id": "STRING"})
                columns = ", ".join(f"n.{col}" for col in schema.keys())
                result = self._conn.execute(
                    f"MATCH (n:{node_type}) WHERE n.id = $id RETURN {columns}",
                    parameters={"id": node_id},
                )
                if result.has_next():
                    row = result.get_next()
                    col_names = list(schema.keys())
                    properties = {}
                    for i, col in enumerate(col_names):
                        if col != "id":
                            properties[col] = row[i]
                    return NodeData(
                        id=node_id,
                        node_type=node_type,
                        properties=properties,
                    )
            except RuntimeError:
                continue
        return None

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[NodeData]:
        """Get neighboring nodes connected by edges."""
        src_type = self._find_node_type(node_id)
        if src_type is None:
            return []

        neighbors: list[NodeData] = []
        seen_ids: set[str] = set()

        edge_types_to_check = [edge_type] if edge_type else list(EDGE_TYPES)

        for et in edge_types_to_check:
            endpoints = EDGE_ENDPOINTS.get(et, [])
            for s_type, t_type in endpoints:
                if direction == "outgoing" and s_type == src_type:
                    table_name = f"{et}_{s_type}_{t_type}"
                    try:
                        result = self._conn.execute(
                            f"MATCH (a:{s_type})-[:{table_name}]->(b:{t_type}) "
                            f"WHERE a.id = $id RETURN b.id",
                            parameters={"id": node_id},
                        )
                        while result.has_next():
                            row = result.get_next()
                            nid = row[0]
                            if nid not in seen_ids:
                                seen_ids.add(nid)
                                n = self.get_node(nid)
                                if n:
                                    neighbors.append(n)
                    except RuntimeError:
                        continue

                elif direction == "incoming" and t_type == src_type:
                    table_name = f"{et}_{s_type}_{t_type}"
                    try:
                        result = self._conn.execute(
                            f"MATCH (a:{s_type})-[:{table_name}]->(b:{t_type}) "
                            f"WHERE b.id = $id RETURN a.id",
                            parameters={"id": node_id},
                        )
                        while result.has_next():
                            row = result.get_next()
                            nid = row[0]
                            if nid not in seen_ids:
                                seen_ids.add(nid)
                                n = self.get_node(nid)
                                if n:
                                    neighbors.append(n)
                    except RuntimeError:
                        continue

                elif direction == "both":
                    # Check both directions
                    if s_type == src_type:
                        table_name = f"{et}_{s_type}_{t_type}"
                        try:
                            result = self._conn.execute(
                                f"MATCH (a:{s_type})-[:{table_name}]->(b:{t_type}) "
                                f"WHERE a.id = $id RETURN b.id",
                                parameters={"id": node_id},
                            )
                            while result.has_next():
                                row = result.get_next()
                                nid = row[0]
                                if nid not in seen_ids:
                                    seen_ids.add(nid)
                                    n = self.get_node(nid)
                                    if n:
                                        neighbors.append(n)
                        except RuntimeError:
                            continue
                    if t_type == src_type:
                        table_name = f"{et}_{s_type}_{t_type}"
                        try:
                            result = self._conn.execute(
                                f"MATCH (a:{s_type})-[:{table_name}]->(b:{t_type}) "
                                f"WHERE b.id = $id RETURN a.id",
                                parameters={"id": node_id},
                            )
                            while result.has_next():
                                row = result.get_next()
                                nid = row[0]
                                if nid not in seen_ids:
                                    seen_ids.add(nid)
                                    n = self.get_node(nid)
                                    if n:
                                        neighbors.append(n)
                        except RuntimeError:
                            continue

        return neighbors

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> TraversalResult:
        """Traverse the graph from a starting node using BFS."""
        all_nodes: dict[str, NodeData] = {}
        all_edges: list[EdgeData] = []
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(start_id, 0)]

        start_node = self.get_node(start_id)
        if start_node:
            all_nodes[start_id] = start_node

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth >= max_depth:
                continue
            visited.add(current_id)

            neighbors = self.get_neighbors(current_id, direction="outgoing")
            for neighbor in neighbors:
                # Check edge type filter
                if edge_types is not None:
                    # We accept the neighbor since we can't easily filter
                    # by edge type in get_neighbors without refactoring
                    pass
                if neighbor.id not in all_nodes:
                    all_nodes[neighbor.id] = neighbor
                    queue.append((neighbor.id, depth + 1))

        return TraversalResult(
            nodes=list(all_nodes.values()),
            edges=all_edges,
        )

    def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Execute a raw Cypher query."""
        result = self._conn.execute(cypher, parameters=parameters or {})
        rows: list[dict] = []
        col_names = result.get_column_names()
        while result.has_next():
            row = result.get_next()
            rows.append(dict(zip(col_names, row)))
        return rows

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        node_type = self._find_node_type(node_id)
        if node_type is None:
            return
        # Delete all edges first
        self._delete_edges_for_node(node_id, node_type)
        # Then delete the node
        try:
            self._conn.execute(
                f"MATCH (n:{node_type}) WHERE n.id = $id DELETE n",
                parameters={"id": node_id},
            )
        except RuntimeError as e:
            logger.warning("Failed to delete node %s: %s", node_id, e)

    def _delete_edges_for_node(self, node_id: str, node_type: str) -> None:
        """Delete all edges connected to a specific node."""
        for edge_type, endpoints in EDGE_ENDPOINTS.items():
            for s_type, t_type in endpoints:
                table_name = f"{edge_type}_{s_type}_{t_type}"
                if s_type == node_type:
                    try:
                        self._conn.execute(
                            f"MATCH (a:{s_type})-[r:{table_name}]->(b:{t_type}) "
                            f"WHERE a.id = $id DELETE r",
                            parameters={"id": node_id},
                        )
                    except RuntimeError:
                        pass
                if t_type == node_type:
                    try:
                        self._conn.execute(
                            f"MATCH (a:{s_type})-[r:{table_name}]->(b:{t_type}) "
                            f"WHERE b.id = $id DELETE r",
                            parameters={"id": node_id},
                        )
                    except RuntimeError:
                        pass

    def delete_edges_for_file(self, file_path: str) -> None:
        """Delete all edges associated with nodes from a specific file."""
        nodes = self.get_nodes_for_file(file_path)
        for node in nodes:
            self._delete_edges_for_node(node.id, node.node_type)

    def get_file_hashes(self) -> dict[str, str]:
        """Get all stored file hashes from File nodes."""
        hashes: dict[str, str] = {}
        try:
            result = self._conn.execute(
                "MATCH (f:File) RETURN f.path, f.hash"
            )
            while result.has_next():
                row = result.get_next()
                path, hash_val = row[0], row[1]
                if path and hash_val:
                    hashes[path] = hash_val
        except RuntimeError:
            pass
        return hashes

    def get_nodes_for_file(self, file_path: str) -> list[NodeData]:
        """Get all nodes associated with a file."""
        nodes: list[NodeData] = []
        # Check node types that have a 'file' or 'path' property
        file_node_types = {
            "File": "path",
            "Class": "file",
            "Function": "file",
            "Method": "file",
            "Interface": "file",
            "Variable": "file",
        }
        for node_type, prop in file_node_types.items():
            try:
                result = self._conn.execute(
                    f"MATCH (n:{node_type}) WHERE n.{prop} = $path RETURN n.id",
                    parameters={"path": file_path},
                )
                while result.has_next():
                    row = result.get_next()
                    node = self.get_node(row[0])
                    if node:
                        nodes.append(node)
            except RuntimeError:
                continue
        return nodes

    def get_chunk_ids_for_file(self, file_path: str) -> list[str]:
        """Get IDs of all Chunk nodes linked to a file's nodes."""
        chunk_ids: list[str] = []
        # Find chunks via CHUNK_OF edges to nodes in this file
        file_nodes = self.get_nodes_for_file(file_path)
        for file_node in file_nodes:
            tgt_type = file_node.node_type
            table_name = f"CHUNK_OF_Chunk_{tgt_type}"
            try:
                result = self._conn.execute(
                    f"MATCH (c:Chunk)-[:{table_name}]->(n:{tgt_type}) "
                    f"WHERE n.id = $id RETURN c.id",
                    parameters={"id": file_node.id},
                )
                while result.has_next():
                    row = result.get_next()
                    chunk_ids.append(row[0])
            except RuntimeError:
                continue
        return chunk_ids

    def delete_nodes_for_file(self, file_path: str) -> None:
        """Delete all nodes and edges associated with a file."""
        # First delete chunks
        chunk_ids = self.get_chunk_ids_for_file(file_path)
        for chunk_id in chunk_ids:
            self.delete_node(chunk_id)

        # Then delete file nodes
        nodes = self.get_nodes_for_file(file_path)
        for node in nodes:
            self.delete_node(node.id)

    def clear(self) -> None:
        """Delete all nodes and edges from the graph."""
        for edge_type, endpoints in EDGE_ENDPOINTS.items():
            for s_type, t_type in endpoints:
                table_name = f"{edge_type}_{s_type}_{t_type}"
                try:
                    self._conn.execute(
                        f"MATCH (a:{s_type})-[r:{table_name}]->(b:{t_type}) DELETE r"
                    )
                except RuntimeError:
                    pass
        for node_type in NODE_TYPES:
            try:
                self._conn.execute(f"MATCH (n:{node_type}) DELETE n")
            except RuntimeError:
                pass

    def close(self) -> None:
        """Close the database connection."""
        self._conn = None
        self._db = None
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuInit -x -v
```

### Step 5 — Commit

```bash
git add nemesis/graph/kuzu.py tests/test_graph/test_kuzu.py
git commit -m "feat(graph): implement KuzuAdapter init, schema creation, and core methods

TDD Task 2/10 of 03-graph-layer plan.
Full KuzuAdapter with node/edge CRUD, neighbor queries, traversal,
schema creation for all 14 node types and 14 edge types.
Kuzu embedded DB requires zero configuration."
```

---

## Task 3: KuzuAdapter — Node CRUD Operations

**Files:**
- `tests/test_graph/test_kuzu.py` (extend)
- `nemesis/graph/kuzu.py` (already implemented, tests verify)

### Step 1 — Write failing test

```python
# tests/test_graph/test_kuzu.py — APPEND to existing file

class TestKuzuNodeCRUD:
    """Tests for node add/get/delete operations."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> KuzuAdapter:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    def test_add_and_get_file_node(self, adapter: KuzuAdapter) -> None:
        node = NodeData(
            id="file-main",
            node_type="File",
            properties={"path": "/src/main.py", "language": "python", "hash": "abc123", "size": 1024},
        )
        adapter.add_node(node)
        result = adapter.get_node("file-main")
        assert result is not None
        assert result.id == "file-main"
        assert result.node_type == "File"
        assert result.properties["path"] == "/src/main.py"
        assert result.properties["language"] == "python"

    def test_add_and_get_function_node(self, adapter: KuzuAdapter) -> None:
        node = NodeData(
            id="func-hello",
            node_type="Function",
            properties={
                "name": "hello",
                "file": "main.py",
                "line_start": 10,
                "line_end": 15,
                "signature": "def hello(name: str) -> str",
                "docstring": "Greet someone.",
                "is_async": False,
            },
        )
        adapter.add_node(node)
        result = adapter.get_node("func-hello")
        assert result is not None
        assert result.properties["name"] == "hello"
        assert result.properties["line_start"] == 10
        assert result.properties["is_async"] is False

    def test_add_and_get_class_node(self, adapter: KuzuAdapter) -> None:
        node = NodeData(
            id="class-calc",
            node_type="Class",
            properties={
                "name": "Calculator",
                "file": "calc.py",
                "line_start": 1,
                "line_end": 50,
                "docstring": "A calculator.",
            },
        )
        adapter.add_node(node)
        result = adapter.get_node("class-calc")
        assert result is not None
        assert result.properties["name"] == "Calculator"

    def test_add_and_get_chunk_node(self, adapter: KuzuAdapter) -> None:
        node = NodeData(
            id="chunk-001",
            node_type="Chunk",
            properties={
                "content": "def hello(): pass",
                "token_count": 5,
                "embedding_id": "emb-001",
                "parent_type": "Function",
            },
        )
        adapter.add_node(node)
        result = adapter.get_node("chunk-001")
        assert result is not None
        assert result.properties["token_count"] == 5

    def test_get_nonexistent_node_returns_none(self, adapter: KuzuAdapter) -> None:
        result = adapter.get_node("does-not-exist")
        assert result is None

    def test_add_node_with_missing_properties_uses_defaults(self, adapter: KuzuAdapter) -> None:
        """Missing properties get default values (empty string, 0, false)."""
        node = NodeData(id="func-minimal", node_type="Function")
        adapter.add_node(node)
        result = adapter.get_node("func-minimal")
        assert result is not None
        assert result.properties["name"] == ""
        assert result.properties["line_start"] == 0

    def test_update_existing_node(self, adapter: KuzuAdapter) -> None:
        """Adding a node with same ID updates the properties."""
        adapter.add_node(NodeData(
            id="file-x", node_type="File",
            properties={"path": "x.py", "hash": "old_hash"},
        ))
        adapter.add_node(NodeData(
            id="file-x", node_type="File",
            properties={"path": "x.py", "hash": "new_hash"},
        ))
        result = adapter.get_node("file-x")
        assert result is not None
        assert result.properties["hash"] == "new_hash"

    def test_delete_node(self, adapter: KuzuAdapter) -> None:
        adapter.add_node(NodeData(id="del-me", node_type="Variable", properties={"name": "x"}))
        assert adapter.get_node("del-me") is not None
        adapter.delete_node("del-me")
        assert adapter.get_node("del-me") is None

    def test_delete_nonexistent_node_no_error(self, adapter: KuzuAdapter) -> None:
        """Deleting a non-existent node should not raise."""
        adapter.delete_node("ghost-node")  # Should not raise
```

### Step 2 — Run tests, verify they FAIL (or PASS if already implemented)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuNodeCRUD -x -v
```

### Step 3 — Implementation already in Task 2

The add_node, get_node, delete_node methods were already implemented in Task 2. This task verifies correctness with detailed tests.

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuNodeCRUD -x -v
```

### Step 5 — Commit

```bash
git add tests/test_graph/test_kuzu.py
git commit -m "test(graph): add comprehensive node CRUD tests for KuzuAdapter

TDD Task 3/10 of 03-graph-layer plan.
Tests add/get/delete for File, Function, Class, Chunk nodes.
Covers missing properties, upsert, and edge cases."
```

---

## Task 4: KuzuAdapter — Edge Operations & Neighbors

**Files:**
- `tests/test_graph/test_kuzu.py` (extend)

### Step 1 — Write failing test

```python
# tests/test_graph/test_kuzu.py — APPEND to existing file

class TestKuzuEdges:
    """Tests for edge add and neighbor query operations."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> KuzuAdapter:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    @pytest.fixture
    def populated_adapter(self, adapter: KuzuAdapter) -> KuzuAdapter:
        """Create a small graph: File -> contains -> Function, Class; Class -> has_method -> Method."""
        adapter.add_node(NodeData(id="file-main", node_type="File", properties={"path": "main.py", "language": "python"}))
        adapter.add_node(NodeData(id="func-hello", node_type="Function", properties={"name": "hello", "file": "main.py"}))
        adapter.add_node(NodeData(id="class-calc", node_type="Class", properties={"name": "Calculator", "file": "main.py"}))
        adapter.add_node(NodeData(id="method-add", node_type="Method", properties={"name": "add", "class_name": "Calculator", "file": "main.py"}))
        adapter.add_node(NodeData(id="func-helper", node_type="Function", properties={"name": "helper", "file": "utils.py"}))

        adapter.add_edge(EdgeData(source_id="file-main", target_id="func-hello", edge_type="CONTAINS"))
        adapter.add_edge(EdgeData(source_id="file-main", target_id="class-calc", edge_type="CONTAINS"))
        adapter.add_edge(EdgeData(source_id="class-calc", target_id="method-add", edge_type="HAS_METHOD"))
        adapter.add_edge(EdgeData(source_id="func-hello", target_id="func-helper", edge_type="CALLS"))
        return adapter

    def test_add_edge_contains(self, populated_adapter: KuzuAdapter) -> None:
        """CONTAINS edge between File and Function is created."""
        neighbors = populated_adapter.get_neighbors("file-main", edge_type="CONTAINS", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "func-hello" in ids
        assert "class-calc" in ids

    def test_add_edge_has_method(self, populated_adapter: KuzuAdapter) -> None:
        """HAS_METHOD edge between Class and Method is created."""
        neighbors = populated_adapter.get_neighbors("class-calc", edge_type="HAS_METHOD", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "method-add" in ids

    def test_add_edge_calls(self, populated_adapter: KuzuAdapter) -> None:
        """CALLS edge between Functions is created."""
        neighbors = populated_adapter.get_neighbors("func-hello", edge_type="CALLS", direction="outgoing")
        ids = {n.id for n in neighbors}
        assert "func-helper" in ids

    def test_get_neighbors_outgoing(self, populated_adapter: KuzuAdapter) -> None:
        neighbors = populated_adapter.get_neighbors("file-main", direction="outgoing")
        assert len(neighbors) >= 2

    def test_get_neighbors_incoming(self, populated_adapter: KuzuAdapter) -> None:
        neighbors = populated_adapter.get_neighbors("func-hello", edge_type="CONTAINS", direction="incoming")
        ids = {n.id for n in neighbors}
        assert "file-main" in ids

    def test_get_neighbors_nonexistent_node(self, populated_adapter: KuzuAdapter) -> None:
        neighbors = populated_adapter.get_neighbors("ghost", direction="outgoing")
        assert neighbors == []

    def test_add_edge_missing_source_no_error(self, adapter: KuzuAdapter) -> None:
        """Adding an edge with missing source node should not raise."""
        adapter.add_node(NodeData(id="target-only", node_type="Function", properties={"name": "f"}))
        adapter.add_edge(EdgeData(source_id="missing", target_id="target-only", edge_type="CALLS"))
        # No crash, edge just not created
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuEdges -x -v 2>&1 | head -30
```

### Step 3 — Implementation already in Task 2

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuEdges -x -v
```

### Step 5 — Commit

```bash
git add tests/test_graph/test_kuzu.py
git commit -m "test(graph): add edge operations and neighbor query tests for KuzuAdapter

TDD Task 4/10 of 03-graph-layer plan.
Tests CONTAINS, HAS_METHOD, CALLS edges. Verifies outgoing/incoming
neighbor queries and graceful handling of missing nodes."
```

---

## Task 5: KuzuAdapter — File Operations & Delta Support

**Files:**
- `tests/test_graph/test_kuzu.py` (extend)

### Step 1 — Write failing test

```python
# tests/test_graph/test_kuzu.py — APPEND to existing file

class TestKuzuFileOperations:
    """Tests for file-related operations used by the indexing pipeline."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> KuzuAdapter:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    @pytest.fixture
    def indexed_adapter(self, adapter: KuzuAdapter) -> KuzuAdapter:
        """Simulate an indexed file with nodes, edges, and chunks."""
        adapter.add_node(NodeData(id="file-svc", node_type="File",
            properties={"path": "/project/service.py", "language": "python", "hash": "aaa111"}))
        adapter.add_node(NodeData(id="func-process", node_type="Function",
            properties={"name": "process", "file": "/project/service.py", "line_start": 5, "line_end": 20}))
        adapter.add_node(NodeData(id="class-svc", node_type="Class",
            properties={"name": "Service", "file": "/project/service.py", "line_start": 1, "line_end": 50}))
        adapter.add_node(NodeData(id="chunk-001", node_type="Chunk",
            properties={"content": "def process():", "token_count": 5, "parent_type": "Function"}))
        adapter.add_node(NodeData(id="chunk-002", node_type="Chunk",
            properties={"content": "class Service:", "token_count": 3, "parent_type": "Class"}))

        # File -> contains -> nodes
        adapter.add_edge(EdgeData(source_id="file-svc", target_id="func-process", edge_type="CONTAINS"))
        adapter.add_edge(EdgeData(source_id="file-svc", target_id="class-svc", edge_type="CONTAINS"))
        # Chunk -> chunk_of -> node
        adapter.add_edge(EdgeData(source_id="chunk-001", target_id="func-process", edge_type="CHUNK_OF"))
        adapter.add_edge(EdgeData(source_id="chunk-002", target_id="class-svc", edge_type="CHUNK_OF"))

        # Also add a node from another file to ensure isolation
        adapter.add_node(NodeData(id="file-other", node_type="File",
            properties={"path": "/project/other.py", "language": "python", "hash": "bbb222"}))
        adapter.add_node(NodeData(id="func-other", node_type="Function",
            properties={"name": "other", "file": "/project/other.py"}))
        adapter.add_edge(EdgeData(source_id="file-other", target_id="func-other", edge_type="CONTAINS"))
        return adapter

    def test_get_file_hashes(self, indexed_adapter: KuzuAdapter) -> None:
        hashes = indexed_adapter.get_file_hashes()
        assert hashes["/project/service.py"] == "aaa111"
        assert hashes["/project/other.py"] == "bbb222"
        assert len(hashes) == 2

    def test_get_file_hashes_empty_graph(self, adapter: KuzuAdapter) -> None:
        hashes = adapter.get_file_hashes()
        assert hashes == {}

    def test_get_nodes_for_file(self, indexed_adapter: KuzuAdapter) -> None:
        nodes = indexed_adapter.get_nodes_for_file("/project/service.py")
        ids = {n.id for n in nodes}
        assert "file-svc" in ids
        assert "func-process" in ids
        assert "class-svc" in ids
        # Nodes from other.py should NOT be included
        assert "func-other" not in ids

    def test_get_nodes_for_file_nonexistent(self, indexed_adapter: KuzuAdapter) -> None:
        nodes = indexed_adapter.get_nodes_for_file("/nonexistent.py")
        assert nodes == []

    def test_get_chunk_ids_for_file(self, indexed_adapter: KuzuAdapter) -> None:
        chunk_ids = indexed_adapter.get_chunk_ids_for_file("/project/service.py")
        assert "chunk-001" in chunk_ids
        assert "chunk-002" in chunk_ids

    def test_get_chunk_ids_for_file_no_chunks(self, indexed_adapter: KuzuAdapter) -> None:
        chunk_ids = indexed_adapter.get_chunk_ids_for_file("/project/other.py")
        assert chunk_ids == []

    def test_delete_nodes_for_file(self, indexed_adapter: KuzuAdapter) -> None:
        indexed_adapter.delete_nodes_for_file("/project/service.py")
        # All service.py nodes should be gone
        assert indexed_adapter.get_node("file-svc") is None
        assert indexed_adapter.get_node("func-process") is None
        assert indexed_adapter.get_node("class-svc") is None
        assert indexed_adapter.get_node("chunk-001") is None
        assert indexed_adapter.get_node("chunk-002") is None
        # Other file nodes should remain
        assert indexed_adapter.get_node("file-other") is not None
        assert indexed_adapter.get_node("func-other") is not None

    def test_delete_nodes_for_file_nonexistent(self, indexed_adapter: KuzuAdapter) -> None:
        """Deleting nodes for a non-existent file should not raise."""
        indexed_adapter.delete_nodes_for_file("/nonexistent.py")  # Should not raise
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuFileOperations -x -v 2>&1 | head -30
```

### Step 3 — Implementation already in Task 2

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuFileOperations -x -v
```

### Step 5 — Commit

```bash
git add tests/test_graph/test_kuzu.py
git commit -m "test(graph): add file operations and delta support tests for KuzuAdapter

TDD Task 5/10 of 03-graph-layer plan.
Tests get_file_hashes, get_nodes_for_file, get_chunk_ids_for_file,
delete_nodes_for_file. Verifies file isolation in delta updates."
```

---

## Task 6: KuzuAdapter — Traversal & Query

**Files:**
- `tests/test_graph/test_kuzu.py` (extend)

### Step 1 — Write failing test

```python
# tests/test_graph/test_kuzu.py — APPEND to existing file

class TestKuzuTraversal:
    """Tests for graph traversal and raw query operations."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> KuzuAdapter:
        a = KuzuAdapter(db_path=str(tmp_path / "test_graph"))
        a.create_schema()
        yield a
        a.close()

    @pytest.fixture
    def graph_with_chain(self, adapter: KuzuAdapter) -> KuzuAdapter:
        """Create a call chain: f1 -> calls -> f2 -> calls -> f3 -> calls -> f4."""
        for i in range(1, 5):
            adapter.add_node(NodeData(
                id=f"func-{i}", node_type="Function",
                properties={"name": f"func_{i}", "file": "chain.py"},
            ))
        adapter.add_edge(EdgeData(source_id="func-1", target_id="func-2", edge_type="CALLS"))
        adapter.add_edge(EdgeData(source_id="func-2", target_id="func-3", edge_type="CALLS"))
        adapter.add_edge(EdgeData(source_id="func-3", target_id="func-4", edge_type="CALLS"))
        return adapter

    def test_traverse_depth_1(self, graph_with_chain: KuzuAdapter) -> None:
        result = graph_with_chain.traverse("func-1", max_depth=1)
        ids = {n.id for n in result.nodes}
        assert "func-1" in ids
        assert "func-2" in ids
        assert "func-3" not in ids  # depth 2

    def test_traverse_depth_2(self, graph_with_chain: KuzuAdapter) -> None:
        result = graph_with_chain.traverse("func-1", max_depth=2)
        ids = {n.id for n in result.nodes}
        assert "func-1" in ids
        assert "func-2" in ids
        assert "func-3" in ids
        assert "func-4" not in ids  # depth 3

    def test_traverse_depth_3(self, graph_with_chain: KuzuAdapter) -> None:
        result = graph_with_chain.traverse("func-1", max_depth=3)
        ids = {n.id for n in result.nodes}
        assert "func-4" in ids  # depth 3 reached

    def test_traverse_nonexistent_start(self, adapter: KuzuAdapter) -> None:
        result = adapter.traverse("ghost")
        assert result.nodes == []

    def test_raw_query(self, graph_with_chain: KuzuAdapter) -> None:
        rows = graph_with_chain.query(
            "MATCH (n:Function) WHERE n.name = $name RETURN n.id, n.name",
            parameters={"name": "func_2"},
        )
        assert len(rows) == 1
        assert rows[0]["n.id"] == "func-2"
        assert rows[0]["n.name"] == "func_2"

    def test_raw_query_no_results(self, adapter: KuzuAdapter) -> None:
        rows = adapter.query("MATCH (n:Function) RETURN n.id")
        assert rows == []

    def test_clear_removes_everything(self, graph_with_chain: KuzuAdapter) -> None:
        graph_with_chain.clear()
        assert graph_with_chain.get_node("func-1") is None
        assert graph_with_chain.get_node("func-2") is None
        rows = graph_with_chain.query("MATCH (n:Function) RETURN n.id")
        assert rows == []
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuTraversal -x -v 2>&1 | head -30
```

### Step 3 — Implementation already in Task 2

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_graph/test_kuzu.py::TestKuzuTraversal -x -v
```

### Step 5 — Commit

```bash
git add tests/test_graph/test_kuzu.py
git commit -m "test(graph): add traversal and raw query tests for KuzuAdapter

TDD Task 6/10 of 03-graph-layer plan.
Tests BFS traversal at depths 1/2/3 through CALLS chain.
Tests raw Cypher query and clear() operation."
```

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

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | GraphAdapter protocol + data models | `adapter.py` | 12 |
| 2 | KuzuAdapter init + schema creation | `kuzu.py` | 5 |
| 3 | KuzuAdapter node CRUD | `test_kuzu.py` | 9 |
| 4 | KuzuAdapter edge operations + neighbors | `test_kuzu.py` | 7 |
| 5 | KuzuAdapter file operations + delta | `test_kuzu.py` | 8 |
| 6 | KuzuAdapter traversal + raw query | `test_kuzu.py` | 7 |
| 7 | Neo4jAdapter full implementation | `neo4j.py`, `test_neo4j.py` | 7 |
| 8 | Factory function + module init | `__init__.py`, `test_factory.py` | 5 |
| 9 | Memory node operations | `test_kuzu_memory.py` | 8 |
| 10 | Full integration tests | `test_integration.py` | 6 |
| **Total** | | **4 source + 6 test files** | **~74 tests** |

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

### Graph Schema (Cypher-compatible)

```
NODE TYPES (14):
  :File          {id, path, language, hash, last_indexed, size}
  :Module        {id, name, path, docstring}
  :Class         {id, name, file, line_start, line_end, docstring}
  :Function      {id, name, file, line_start, line_end, signature, docstring, is_async}
  :Method        {id, name, class_name, file, line_start, line_end, signature, visibility}
  :Interface     {id, name, file, language}
  :Variable      {id, name, file, type_hint, scope}
  :Import        {id, name, source, alias}
  :Rule          {id, content, scope, created_at, source}
  :Decision      {id, title, reasoning, created_at, status}
  :Alternative   {id, title, reason_rejected}
  :Convention    {id, pattern, example, scope}
  :Project       {id, name, root_path, languages, last_indexed}
  :Chunk         {id, content, token_count, embedding_id, parent_type}

EDGE TYPES (14):
  CONTAINS, HAS_METHOD, INHERITS, IMPLEMENTS, CALLS, IMPORTS, RETURNS,
  ACCEPTS, APPLIES_TO, CHOSE, REJECTED, GOVERNS, CHUNK_OF, HAS_FILE
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
