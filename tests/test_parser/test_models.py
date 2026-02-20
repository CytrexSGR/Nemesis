"""Tests for parser Python data models."""

from nemesis.parser.models import (
    CodeEdge,
    CodeNode,
    EdgeKind,
    ExtractionResult,
    NodeKind,
)


class TestNodeKind:
    def test_all_kinds_exist(self) -> None:
        kinds = [
            NodeKind.FILE,
            NodeKind.MODULE,
            NodeKind.CLASS,
            NodeKind.FUNCTION,
            NodeKind.METHOD,
            NodeKind.INTERFACE,
            NodeKind.VARIABLE,
            NodeKind.IMPORT,
        ]
        assert len(kinds) == 8

    def test_kind_values(self) -> None:
        assert NodeKind.FILE == "File"
        assert NodeKind.FUNCTION == "Function"


class TestEdgeKind:
    def test_all_kinds_exist(self) -> None:
        kinds = [
            EdgeKind.CONTAINS,
            EdgeKind.HAS_METHOD,
            EdgeKind.INHERITS,
            EdgeKind.IMPLEMENTS,
            EdgeKind.CALLS,
            EdgeKind.IMPORTS,
            EdgeKind.RETURNS,
            EdgeKind.ACCEPTS,
        ]
        assert len(kinds) == 8

    def test_kind_values(self) -> None:
        assert EdgeKind.CONTAINS == "CONTAINS"
        assert EdgeKind.HAS_METHOD == "HAS_METHOD"


class TestCodeNode:
    def test_creation(self) -> None:
        node = CodeNode(
            id="func:test.py:hello:1",
            kind=NodeKind.FUNCTION,
            name="hello",
            file="test.py",
            line_start=1,
            line_end=3,
            language="python",
        )
        assert node.name == "hello"
        assert node.kind == NodeKind.FUNCTION
        assert node.docstring is None
        assert node.is_async is False

    def test_from_dict(self) -> None:
        data = {
            "id": "class:t.py:Foo:1",
            "kind": "Class",
            "name": "Foo",
            "file": "t.py",
            "line_start": 1,
            "line_end": 5,
            "language": "python",
            "docstring": "A foo.",
            "is_async": False,
        }
        node = CodeNode.from_dict(data)
        assert node.name == "Foo"
        assert node.docstring == "A foo."


class TestCodeEdge:
    def test_creation(self) -> None:
        edge = CodeEdge(
            source_id="file:test.py",
            target_id="func:test.py:hello:1",
            kind=EdgeKind.CONTAINS,
            file="test.py",
        )
        assert edge.kind == EdgeKind.CONTAINS

    def test_from_dict(self) -> None:
        data = {
            "source_id": "a",
            "target_id": "b",
            "kind": "CONTAINS",
            "file": "t.py",
        }
        edge = CodeEdge.from_dict(data)
        assert edge.kind == EdgeKind.CONTAINS


class TestExtractionResult:
    def test_creation(self) -> None:
        result = ExtractionResult(
            file="test.py",
            language="python",
            nodes=[],
            edges=[],
        )
        assert result.file == "test.py"
        assert len(result.nodes) == 0

    def test_from_dict(self) -> None:
        data = {
            "file": "t.py",
            "language": "python",
            "nodes": [
                {
                    "id": "file:t.py",
                    "kind": "File",
                    "name": "t.py",
                    "file": "t.py",
                    "line_start": 1,
                    "line_end": 1,
                    "language": "python",
                    "is_async": False,
                },
            ],
            "edges": [],
        }
        result = ExtractionResult.from_dict(data)
        assert len(result.nodes) == 1
        assert result.nodes[0].kind == NodeKind.FILE

    def test_node_count_property(self) -> None:
        node = CodeNode(
            id="x",
            kind=NodeKind.FILE,
            name="x",
            file="x",
            line_start=1,
            line_end=1,
            language="python",
        )
        result = ExtractionResult(file="x", language="python", nodes=[node], edges=[])
        assert result.node_count == 1
        assert result.edge_count == 0
