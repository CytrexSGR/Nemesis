"""Tests for the Python bridge to nemesis-parse Rust extension."""

from pathlib import Path

import pytest

from nemesis.parser.bridge import ParserBridge, ParserError
from nemesis.parser.models import EdgeKind, ExtractionResult, NodeKind


class TestParserBridgeInit:
    def test_bridge_creation(self) -> None:
        bridge = ParserBridge()
        assert bridge is not None

    def test_bridge_has_native_module(self) -> None:
        bridge = ParserBridge()
        assert bridge.native_available is True or bridge.native_available is False

    def test_supported_languages(self) -> None:
        bridge = ParserBridge()
        langs = bridge.supported_languages()
        assert "python" in langs


class TestParserBridgeDetect:
    def test_detect_python(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("app.py") == "python"

    def test_detect_typescript(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("app.ts") == "typescript"

    def test_detect_rust(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("main.rs") == "rust"

    def test_detect_unsupported(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.detect_language("file.java")


class TestParserBridgeParseString:
    def test_parse_python_string(self) -> None:
        bridge = ParserBridge()
        result = bridge.parse_string("def hello(): pass\n", "python", "test.py")
        assert isinstance(result, ExtractionResult)
        assert result.file == "test.py"
        assert result.language == "python"
        assert result.node_count >= 2  # File + Function

    def test_parse_python_class(self) -> None:
        bridge = ParserBridge()
        src = "class Foo:\n    def bar(self):\n        pass\n"
        result = bridge.parse_string(src, "python", "test.py")
        class_nodes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        method_nodes = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        assert len(class_nodes) == 1
        assert len(method_nodes) == 1
        assert class_nodes[0].name == "Foo"

    def test_parse_typescript_string(self) -> None:
        bridge = ParserBridge()
        src = "function greet(name: string): void { console.log(name); }\n"
        result = bridge.parse_string(src, "typescript", "test.ts")
        assert result.language == "typescript"
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 1

    def test_parse_rust_string(self) -> None:
        bridge = ParserBridge()
        src = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        result = bridge.parse_string(src, "rust", "lib.rs")
        assert result.language == "rust"
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 1

    def test_parse_empty_source(self) -> None:
        bridge = ParserBridge()
        result = bridge.parse_string("", "python", "empty.py")
        assert result.node_count >= 1  # At least File node

    def test_parse_invalid_language_raises(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.parse_string("code", "java", "test.java")


class TestParserBridgeParseFile:
    def test_parse_python_file(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        assert result.language == "python"
        class_nodes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        assert len(class_nodes) >= 1
        assert any(n.name == "Calculator" for n in class_nodes)

    def test_parse_nonexistent_file_raises(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.parse_file("/nonexistent/path/file.py")

    def test_parse_file_extracts_methods(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "add" in method_names
        assert "subtract" in method_names

    def test_parse_file_extracts_edges(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        contains = [e for e in result.edges if e.kind == EdgeKind.CONTAINS]
        has_method = [e for e in result.edges if e.kind == EdgeKind.HAS_METHOD]
        assert len(contains) >= 1
        assert len(has_method) >= 1


class TestParserBridgeEdgeCases:
    def test_parse_syntax_error_still_produces_partial(self) -> None:
        """Tree-sitter is error-tolerant; partial results are expected."""
        bridge = ParserBridge()
        src = "def broken(:\n    pass\n"
        result = bridge.parse_string(src, "python", "broken.py")
        # Should still produce a File node at minimum
        assert result.node_count >= 1

    def test_parse_large_source(self) -> None:
        bridge = ParserBridge()
        # Generate a file with 100 functions
        lines = []
        for i in range(100):
            lines.append(f"def func_{i}():\n    pass\n")
        src = "\n".join(lines)
        result = bridge.parse_string(src, "python", "big.py")
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 100
