# Graph Layer — Arbeitspaket C2: Kuzu CRUD

> **Arbeitspaket C2** — Teil 2 von 4 des Graph Layer Plans

**Goal:** Kuzu CRUD-Operationen testen und validieren: Node CRUD, Edge Operations, File Operations und Traversal (Tasks 3, 4, 5, 6).

**Architecture:** Alle Tests verifizieren die in Task 2 (C1) implementierte `KuzuAdapter`-Klasse. Die Implementierung ist bereits vorhanden — dieses Paket fuegt umfassende Tests hinzu.

**Tech Stack:** Python 3.11+, Kuzu >= 0.4, pytest

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [03a-graph-foundation](03a-graph-foundation.md) (C1)

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

## Summary

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 3 | KuzuAdapter Node CRUD | `test_kuzu.py` | 9 |
| 4 | KuzuAdapter Edge Operations + Neighbors | `test_kuzu.py` | 7 |
| 5 | KuzuAdapter File Operations + Delta | `test_kuzu.py` | 8 |
| 6 | KuzuAdapter Traversal + Raw Query | `test_kuzu.py` | 7 |
| **Gesamt C2** | | **1 Test-Datei (erweitert)** | **~31 Tests** |

---

**Navigation:**
- Vorheriges Paket: [C1 — Graph Foundation](03a-graph-foundation.md)
- Nachstes Paket: [C3 — Neo4j & Memory](03c-neo4j-memory.md)
- Gesamtplan: [03-graph-layer.md](03-graph-layer.md)
