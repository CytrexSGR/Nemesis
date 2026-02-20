# Graph Layer — Arbeitspaket C1: Graph Foundation

> **Arbeitspaket C1** — Teil 1 von 4 des Graph Layer Plans

**Goal:** GraphAdapter Protocol definieren und KuzuAdapter mit Schema-Erstellung initialisieren (Tasks 1, 2).

**Architecture:** `GraphAdapter` Protocol defines the common interface. `KuzuAdapter` provides zero-config embedded graph storage (default). Both backends use Cypher-compatible schemas. All graph operations are synchronous since both Kuzu and Neo4j Python drivers are sync.

**Tech Stack:** Python 3.11+, Kuzu >= 0.4, pytest

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

## Summary

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 1 | GraphAdapter Protocol + Data Models | `adapter.py` | 12 |
| 2 | KuzuAdapter Init + Schema Creation | `kuzu.py` | 5 |
| **Gesamt C1** | | **2 Source + 2 Test-Dateien** | **~17 Tests** |

---

**Navigation:**
- Vorheriges Paket: --
- Nachstes Paket: [C2 — Kuzu CRUD](03b-kuzu-crud.md)
- Gesamtplan: [03-graph-layer.md](03-graph-layer.md)
