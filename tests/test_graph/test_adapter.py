"""Tests for GraphAdapter protocol and graph data models."""

from dataclasses import asdict

from nemesis.graph.adapter import (
    EDGE_TYPES,
    NODE_TYPES,
    EdgeData,
    GraphAdapter,
    NodeData,
    TraversalResult,
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
            "File",
            "Module",
            "Class",
            "Function",
            "Method",
            "Interface",
            "Variable",
            "Import",
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
            "CONTAINS",
            "HAS_METHOD",
            "INHERITS",
            "IMPLEMENTS",
            "CALLS",
            "IMPORTS",
            "RETURNS",
            "ACCEPTS",
            "APPLIES_TO",
            "CHOSE",
            "REJECTED",
            "GOVERNS",
            "CHUNK_OF",
            "HAS_FILE",
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
                self,
                node_id: str,
                edge_type: str | None = None,
                direction: str = "outgoing",
            ) -> list[NodeData]: ...
            def traverse(
                self,
                start_id: str,
                edge_types: list[str] | None = None,
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
