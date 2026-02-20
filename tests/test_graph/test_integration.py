"""Integration tests — full graph workflow with a real Kuzu database."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from nemesis.graph import create_graph_adapter
from nemesis.graph.adapter import EdgeData, NodeData


@pytest.fixture()
def adapter(tmp_path: Path):
    """Create a Kuzu adapter with schema and close on teardown."""
    db_path = str(tmp_path / "integration_graph")
    a = create_graph_adapter(backend="kuzu", db_path=db_path, create_schema=True)
    yield a
    a.close()


class TestIndexAndQueryFile:
    """Add File + Class + Methods + Function with edges, query neighbors."""

    def test_index_and_query_file(self, adapter) -> None:
        # Add a File node
        adapter.add_node(
            NodeData(
                id="file:auth.py",
                node_type="File",
                properties={"path": "src/auth.py", "language": "python", "hash": "abc123"},
            )
        )

        # Add a Class node
        adapter.add_node(
            NodeData(
                id="class:AuthService",
                node_type="Class",
                properties={"name": "AuthService", "file": "src/auth.py", "line_start": 10},
            )
        )

        # Add Method nodes
        adapter.add_node(
            NodeData(
                id="method:login",
                node_type="Method",
                properties={
                    "name": "login",
                    "class_name": "AuthService",
                    "file": "src/auth.py",
                    "line_start": 15,
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="method:logout",
                node_type="Method",
                properties={
                    "name": "logout",
                    "class_name": "AuthService",
                    "file": "src/auth.py",
                    "line_start": 30,
                },
            )
        )

        # Add a standalone Function
        adapter.add_node(
            NodeData(
                id="func:hash_password",
                node_type="Function",
                properties={
                    "name": "hash_password",
                    "file": "src/auth.py",
                    "line_start": 50,
                },
            )
        )

        # Add edges: File CONTAINS Class, File CONTAINS Function
        adapter.add_edge(
            EdgeData(source_id="file:auth.py", target_id="class:AuthService", edge_type="CONTAINS")
        )
        adapter.add_edge(
            EdgeData(source_id="file:auth.py", target_id="func:hash_password", edge_type="CONTAINS")
        )

        # Class HAS_METHOD
        adapter.add_edge(
            EdgeData(
                source_id="class:AuthService", target_id="method:login", edge_type="HAS_METHOD"
            )
        )
        adapter.add_edge(
            EdgeData(
                source_id="class:AuthService", target_id="method:logout", edge_type="HAS_METHOD"
            )
        )

        # Query outgoing CONTAINS neighbors of the file
        neighbors = adapter.get_neighbors(
            "file:auth.py", edge_type="CONTAINS", direction="outgoing"
        )
        neighbor_ids = {n.id for n in neighbors}
        assert "class:AuthService" in neighbor_ids
        assert "func:hash_password" in neighbor_ids

        # Query HAS_METHOD neighbors of the class
        methods = adapter.get_neighbors(
            "class:AuthService", edge_type="HAS_METHOD", direction="outgoing"
        )
        method_ids = {m.id for m in methods}
        assert "method:login" in method_ids
        assert "method:logout" in method_ids
        assert len(methods) == 2


class TestDeltaUpdateWorkflow:
    """test_delta_update_workflow: add nodes, verify hashes, delete, re-add, verify."""

    def test_delta_update_workflow(self, adapter) -> None:
        # Initial indexing
        adapter.add_node(
            NodeData(
                id="file:utils.py",
                node_type="File",
                properties={"path": "src/utils.py", "language": "python", "hash": "v1hash"},
            )
        )
        adapter.add_node(
            NodeData(
                id="func:old_helper",
                node_type="Function",
                properties={"name": "old_helper", "file": "src/utils.py", "line_start": 1},
            )
        )
        adapter.add_edge(
            EdgeData(source_id="file:utils.py", target_id="func:old_helper", edge_type="CONTAINS")
        )

        # Verify hash is stored
        hashes = adapter.get_file_hashes()
        assert hashes.get("src/utils.py") == "v1hash"

        # Verify nodes for file
        file_nodes = adapter.get_nodes_for_file("src/utils.py")
        file_node_ids = {n.id for n in file_nodes}
        assert "file:utils.py" in file_node_ids
        assert "func:old_helper" in file_node_ids

        # Delta update: file changed, delete old nodes
        adapter.delete_nodes_for_file("src/utils.py")

        # Verify all old nodes are gone
        assert adapter.get_node("file:utils.py") is None
        assert adapter.get_node("func:old_helper") is None

        # Add new version
        adapter.add_node(
            NodeData(
                id="file:utils.py",
                node_type="File",
                properties={"path": "src/utils.py", "language": "python", "hash": "v2hash"},
            )
        )
        adapter.add_node(
            NodeData(
                id="func:new_helper",
                node_type="Function",
                properties={"name": "new_helper", "file": "src/utils.py", "line_start": 1},
            )
        )
        adapter.add_edge(
            EdgeData(source_id="file:utils.py", target_id="func:new_helper", edge_type="CONTAINS")
        )

        # Verify new version
        hashes = adapter.get_file_hashes()
        assert hashes.get("src/utils.py") == "v2hash"

        new_nodes = adapter.get_nodes_for_file("src/utils.py")
        new_ids = {n.id for n in new_nodes}
        assert "func:new_helper" in new_ids
        assert "func:old_helper" not in new_ids


class TestTraversalWithRealisticGraph:
    """test_traversal_with_realistic_graph: chain via CALLS, traverse depth=2."""

    def test_traversal_with_realistic_graph(self, adapter) -> None:
        # Create a call chain: authenticate -> validate_token -> db_lookup
        adapter.add_node(
            NodeData(
                id="func:authenticate",
                node_type="Function",
                properties={"name": "authenticate", "file": "src/auth.py", "line_start": 1},
            )
        )
        adapter.add_node(
            NodeData(
                id="func:validate_token",
                node_type="Function",
                properties={"name": "validate_token", "file": "src/auth.py", "line_start": 20},
            )
        )
        adapter.add_node(
            NodeData(
                id="func:db_lookup",
                node_type="Function",
                properties={"name": "db_lookup", "file": "src/db.py", "line_start": 5},
            )
        )

        # authenticate CALLS validate_token
        adapter.add_edge(
            EdgeData(
                source_id="func:authenticate",
                target_id="func:validate_token",
                edge_type="CALLS",
            )
        )
        # validate_token CALLS db_lookup
        adapter.add_edge(
            EdgeData(
                source_id="func:validate_token",
                target_id="func:db_lookup",
                edge_type="CALLS",
            )
        )

        # Traverse from authenticate with depth=2
        result = adapter.traverse(start_id="func:authenticate", edge_types=["CALLS"], max_depth=2)

        traversed_ids = {n.id for n in result.nodes}
        assert "func:authenticate" in traversed_ids
        assert "func:validate_token" in traversed_ids
        # depth=2 means: depth 0 = authenticate (visited), depth 1 = validate_token (visited),
        # at depth 1 we look for neighbors -> db_lookup added at depth 2,
        # but not visited (depth >= max_depth).
        # The BFS adds db_lookup to all_nodes before checking depth, so it should be present.
        assert "func:db_lookup" in traversed_ids


class TestMemoryIntegration:
    """test_memory_integration: code nodes + Rule node with APPLIES_TO edge."""

    def test_memory_integration(self, adapter) -> None:
        # Add a File and a Class
        adapter.add_node(
            NodeData(
                id="file:models.py",
                node_type="File",
                properties={"path": "src/models.py", "language": "python", "hash": "modelhash"},
            )
        )
        adapter.add_node(
            NodeData(
                id="class:User",
                node_type="Class",
                properties={"name": "User", "file": "src/models.py", "line_start": 5},
            )
        )

        # Add a Rule node
        adapter.add_node(
            NodeData(
                id="rule:no-orm-in-views",
                node_type="Rule",
                properties={
                    "content": "Do not use ORM queries directly in views",
                    "scope": "project",
                    "source": "team-lead",
                },
            )
        )

        # Rule APPLIES_TO File and Class
        adapter.add_edge(
            EdgeData(
                source_id="rule:no-orm-in-views",
                target_id="file:models.py",
                edge_type="APPLIES_TO",
            )
        )
        adapter.add_edge(
            EdgeData(
                source_id="rule:no-orm-in-views",
                target_id="class:User",
                edge_type="APPLIES_TO",
            )
        )

        # Query incoming neighbors of file:models.py (rules that apply to this file)
        incoming = adapter.get_neighbors(
            "file:models.py", edge_type="APPLIES_TO", direction="incoming"
        )
        incoming_ids = {n.id for n in incoming}
        assert "rule:no-orm-in-views" in incoming_ids

        # Verify the rule itself
        rule = adapter.get_node("rule:no-orm-in-views")
        assert rule is not None
        assert rule.node_type == "Rule"
        assert rule.properties["content"] == "Do not use ORM queries directly in views"

        # Query incoming of the class too
        class_incoming = adapter.get_neighbors(
            "class:User", edge_type="APPLIES_TO", direction="incoming"
        )
        class_incoming_ids = {n.id for n in class_incoming}
        assert "rule:no-orm-in-views" in class_incoming_ids


class TestChunkLifecycle:
    """test_chunk_lifecycle: File->Function + Chunks with CHUNK_OF edges."""

    def test_chunk_lifecycle(self, adapter) -> None:
        # Create File and Function
        adapter.add_node(
            NodeData(
                id="file:service.py",
                node_type="File",
                properties={"path": "src/service.py", "language": "python", "hash": "svchash"},
            )
        )
        adapter.add_node(
            NodeData(
                id="func:process",
                node_type="Function",
                properties={"name": "process", "file": "src/service.py", "line_start": 10},
            )
        )
        adapter.add_edge(
            EdgeData(source_id="file:service.py", target_id="func:process", edge_type="CONTAINS")
        )

        # Create Chunk nodes linked to the Function
        adapter.add_node(
            NodeData(
                id="chunk:process:0",
                node_type="Chunk",
                properties={
                    "content": "def process(data):",
                    "token_count": 15,
                    "parent_type": "Function",
                },
            )
        )
        adapter.add_node(
            NodeData(
                id="chunk:process:1",
                node_type="Chunk",
                properties={
                    "content": "    return transform(data)",
                    "token_count": 12,
                    "parent_type": "Function",
                },
            )
        )

        # CHUNK_OF edges
        adapter.add_edge(
            EdgeData(source_id="chunk:process:0", target_id="func:process", edge_type="CHUNK_OF")
        )
        adapter.add_edge(
            EdgeData(source_id="chunk:process:1", target_id="func:process", edge_type="CHUNK_OF")
        )

        # Verify chunk IDs for file
        chunk_ids = adapter.get_chunk_ids_for_file("src/service.py")
        assert "chunk:process:0" in chunk_ids
        assert "chunk:process:1" in chunk_ids
        assert len(chunk_ids) == 2

        # Delete nodes for file should also remove chunks
        adapter.delete_nodes_for_file("src/service.py")

        # Verify everything is gone
        assert adapter.get_node("file:service.py") is None
        assert adapter.get_node("func:process") is None
        assert adapter.get_node("chunk:process:0") is None
        assert adapter.get_node("chunk:process:1") is None

        # Verify no chunks remain for the file
        remaining_chunks = adapter.get_chunk_ids_for_file("src/service.py")
        assert len(remaining_chunks) == 0


class TestClearRemovesAllData:
    """test_clear_removes_all_data: add nodes+edges, clear(), verify everything gone."""

    def test_clear_removes_all_data(self, adapter) -> None:
        # Add nodes of different types
        adapter.add_node(
            NodeData(
                id="file:main.py",
                node_type="File",
                properties={"path": "src/main.py", "language": "python", "hash": "mainhash"},
            )
        )
        adapter.add_node(
            NodeData(
                id="func:main",
                node_type="Function",
                properties={"name": "main", "file": "src/main.py", "line_start": 1},
            )
        )
        adapter.add_node(
            NodeData(
                id="class:App",
                node_type="Class",
                properties={"name": "App", "file": "src/main.py", "line_start": 20},
            )
        )
        adapter.add_node(
            NodeData(
                id="rule:style",
                node_type="Rule",
                properties={"content": "Use black formatter", "scope": "project"},
            )
        )

        # Add edges
        adapter.add_edge(
            EdgeData(source_id="file:main.py", target_id="func:main", edge_type="CONTAINS")
        )
        adapter.add_edge(
            EdgeData(source_id="file:main.py", target_id="class:App", edge_type="CONTAINS")
        )
        adapter.add_edge(
            EdgeData(source_id="rule:style", target_id="file:main.py", edge_type="APPLIES_TO")
        )

        # Verify nodes exist
        assert adapter.get_node("file:main.py") is not None
        assert adapter.get_node("func:main") is not None
        assert adapter.get_node("class:App") is not None
        assert adapter.get_node("rule:style") is not None

        # Clear everything
        adapter.clear()

        # Verify all nodes are gone
        assert adapter.get_node("file:main.py") is None
        assert adapter.get_node("func:main") is None
        assert adapter.get_node("class:App") is None
        assert adapter.get_node("rule:style") is None

        # Verify no file hashes remain
        assert adapter.get_file_hashes() == {}

        # Verify no neighbors can be found
        assert adapter.get_neighbors("file:main.py") == []
