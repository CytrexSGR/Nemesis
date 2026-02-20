"""Multi-language integration tests for the nemesis-parse Rust extension.

These tests parse real sample files through the Python bridge and verify
that the extraction produces correct nodes, edges, and metadata for
Python, TypeScript, and Rust source files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemesis.parser.bridge import ParserBridge
from nemesis.parser.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def bridge() -> ParserBridge:
    """Create a ParserBridge instance for integration testing."""
    b = ParserBridge()
    if not b.native_available:
        pytest.skip("nemesis-parse native extension not available")
    return b


# ---------------------------------------------------------------------------
# Python integration
# ---------------------------------------------------------------------------


class TestPythonIntegration:
    """Integration tests for complex Python file parsing."""

    def test_parse_complex_python_file(
        self,
        bridge: ParserBridge,
        sample_complex_python: Path,
    ) -> None:
        """Parse the complex Python fixture and verify basic result properties."""
        result = bridge.parse_file(str(sample_complex_python))
        assert result.language == "python"
        assert result.node_count > 5

    def test_extracts_classes_with_inheritance(
        self,
        bridge: ParserBridge,
        sample_complex_python: Path,
    ) -> None:
        """BaseService and UserService classes should both be extracted."""
        result = bridge.parse_file(str(sample_complex_python))
        class_nodes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        class_names = {n.name for n in class_nodes}
        assert "BaseService" in class_names
        assert "UserService" in class_names

    def test_extracts_inherits_edge(
        self,
        bridge: ParserBridge,
        sample_complex_python: Path,
    ) -> None:
        """An INHERITS edge should exist originating from UserService."""
        result = bridge.parse_file(str(sample_complex_python))
        inherits_edges = [e for e in result.edges if e.kind == EdgeKind.INHERITS]
        assert len(inherits_edges) >= 1
        # The source of the inherits edge must reference UserService
        assert any("UserService" in e.source_id for e in inherits_edges)

    def test_extracts_imports(
        self,
        bridge: ParserBridge,
        sample_complex_python: Path,
    ) -> None:
        """At least 2 import nodes should be present (os, pathlib, typing)."""
        result = bridge.parse_file(str(sample_complex_python))
        import_nodes = [n for n in result.nodes if n.kind == NodeKind.IMPORT]
        assert len(import_nodes) >= 2

    def test_extracts_methods_with_visibility(
        self,
        bridge: ParserBridge,
        sample_complex_python: Path,
    ) -> None:
        """get_user should be public, _internal_method should be protected."""
        result = bridge.parse_file(str(sample_complex_python))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_map = {m.name: m for m in methods}

        assert "get_user" in method_map
        assert method_map["get_user"].visibility == "public"

        assert "_internal_method" in method_map
        assert method_map["_internal_method"].visibility == "protected"

    def test_extracts_standalone_functions(
        self,
        bridge: ParserBridge,
        sample_complex_python: Path,
    ) -> None:
        """The create_service factory function should be extracted."""
        result = bridge.parse_file(str(sample_complex_python))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        func_names = {f.name for f in funcs}
        assert "create_service" in func_names


# ---------------------------------------------------------------------------
# TypeScript integration
# ---------------------------------------------------------------------------


class TestTypeScriptIntegration:
    """Integration tests for TypeScript file parsing."""

    def test_parse_typescript_file(
        self,
        bridge: ParserBridge,
        sample_typescript_file: Path,
    ) -> None:
        """Parse the TypeScript fixture and verify basic result properties."""
        result = bridge.parse_file(str(sample_typescript_file))
        assert result.language == "typescript"
        assert result.node_count > 3

    def test_extracts_interface(
        self,
        bridge: ParserBridge,
        sample_typescript_file: Path,
    ) -> None:
        """An Interface node named 'User' should be extracted."""
        result = bridge.parse_file(str(sample_typescript_file))
        ifaces = [n for n in result.nodes if n.kind == NodeKind.INTERFACE]
        iface_names = {n.name for n in ifaces}
        assert "User" in iface_names

    def test_extracts_class_with_methods(
        self,
        bridge: ParserBridge,
        sample_typescript_file: Path,
    ) -> None:
        """UserService class with getUser and addUser methods should be found."""
        result = bridge.parse_file(str(sample_typescript_file))
        classes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        class_names = {n.name for n in classes}
        assert "UserService" in class_names

        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "getUser" in method_names
        assert "addUser" in method_names

    def test_extracts_function(
        self,
        bridge: ParserBridge,
        sample_typescript_file: Path,
    ) -> None:
        """The createApp function should be extracted."""
        result = bridge.parse_file(str(sample_typescript_file))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        func_names = {f.name for f in funcs}
        assert "createApp" in func_names


# ---------------------------------------------------------------------------
# Rust integration
# ---------------------------------------------------------------------------


class TestRustIntegration:
    """Integration tests for Rust file parsing."""

    def test_parse_rust_file(
        self,
        bridge: ParserBridge,
        sample_rust_file: Path,
    ) -> None:
        """Parse the Rust fixture and verify basic result properties."""
        result = bridge.parse_file(str(sample_rust_file))
        assert result.language == "rust"
        assert result.node_count > 3

    def test_extracts_struct_as_class(
        self,
        bridge: ParserBridge,
        sample_rust_file: Path,
    ) -> None:
        """A struct 'Person' should be extracted as NodeKind.CLASS."""
        result = bridge.parse_file(str(sample_rust_file))
        classes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        class_names = {n.name for n in classes}
        assert "Person" in class_names

    def test_extracts_trait_as_interface(
        self,
        bridge: ParserBridge,
        sample_rust_file: Path,
    ) -> None:
        """A trait 'Greetable' should be extracted as NodeKind.INTERFACE."""
        result = bridge.parse_file(str(sample_rust_file))
        ifaces = [n for n in result.nodes if n.kind == NodeKind.INTERFACE]
        iface_names = {n.name for n in ifaces}
        assert "Greetable" in iface_names

    def test_extracts_impl_methods(
        self,
        bridge: ParserBridge,
        sample_rust_file: Path,
    ) -> None:
        """Methods 'new' and 'is_adult' should be extracted from impl Person."""
        result = bridge.parse_file(str(sample_rust_file))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "new" in method_names
        assert "is_adult" in method_names

    def test_extracts_pub_function(
        self,
        bridge: ParserBridge,
        sample_rust_file: Path,
    ) -> None:
        """The create_person function should have visibility=='pub'."""
        result = bridge.parse_file(str(sample_rust_file))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        cp = [f for f in funcs if f.name == "create_person"]
        assert len(cp) == 1
        assert cp[0].visibility == "pub"

    def test_has_method_edges(
        self,
        bridge: ParserBridge,
        sample_rust_file: Path,
    ) -> None:
        """At least 2 HAS_METHOD edges should exist (from impl blocks)."""
        result = bridge.parse_file(str(sample_rust_file))
        hm_edges = [e for e in result.edges if e.kind == EdgeKind.HAS_METHOD]
        assert len(hm_edges) >= 2
