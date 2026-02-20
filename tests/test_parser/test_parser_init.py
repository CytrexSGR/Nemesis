"""Tests for parser subpackage public API."""


def test_parser_exports_bridge() -> None:
    from nemesis.parser import ParserBridge

    assert ParserBridge is not None


def test_parser_exports_error() -> None:
    from nemesis.parser import ParserError

    assert ParserError is not None


def test_parser_exports_models() -> None:
    from nemesis.parser import CodeEdge, CodeNode, ExtractionResult

    assert CodeNode is not None
    assert CodeEdge is not None
    assert ExtractionResult is not None


def test_parser_exports_kinds() -> None:
    from nemesis.parser import EdgeKind, NodeKind

    assert NodeKind.FILE == "File"
    assert EdgeKind.CONTAINS == "CONTAINS"
