"""Tests fuer nemesis.indexer.chunker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nemesis.indexer.chunker import chunk_node


@dataclass
class MockCodeNode:
    id: str
    node_type: str
    name: str
    start_line: int
    end_line: int
    children: list = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


# ---------------------------------------------------------------------------
# Task 3: Small node passthrough chunker
# ---------------------------------------------------------------------------


class TestSmallNodePassthrough:
    def test_small_node_becomes_single_chunk(self):
        """Node unter max_tokens ergibt genau 1 Chunk."""
        node = MockCodeNode(id="func-1", node_type="function", name="foo", start_line=1, end_line=5)
        source = "def foo():\n    return 42\n"
        chunks = chunk_node(node, source, max_tokens=500)
        assert len(chunks) == 1

    def test_chunk_id_format(self):
        """Chunk-ID beginnt mit der Node-ID."""
        node = MockCodeNode(id="cls-7", node_type="class", name="Bar", start_line=10, end_line=20)
        source = "class Bar:\n    pass\n"
        chunks = chunk_node(node, source, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0].id.startswith("cls-7")
        assert chunks[0].id == "cls-7:chunk-0"

    def test_empty_source_returns_empty(self):
        """Leerer Quellcode ergibt 0 Chunks."""
        node = MockCodeNode(id="func-2", node_type="function", name="baz", start_line=1, end_line=1)
        assert chunk_node(node, "", max_tokens=500) == []
        assert chunk_node(node, "   \n\t  ", max_tokens=500) == []

    def test_chunk_file_path_from_node(self):
        """file_path wird korrekt an den Chunk weitergegeben."""
        node = MockCodeNode(id="func-3", node_type="function", name="qux", start_line=1, end_line=3)
        source = "def qux():\n    pass\n"
        fp = Path("/src/main.py")
        chunks = chunk_node(node, source, max_tokens=500, file_path=fp)
        assert len(chunks) == 1
        assert chunks[0].file_path == fp

    def test_chunk_token_count_set(self):
        """token_count ist groesser als 0."""
        node = MockCodeNode(
            id="func-4", node_type="function", name="greet", start_line=1, end_line=3
        )
        source = "def greet(name):\n    print(f'Hello {name}')\n"
        chunks = chunk_node(node, source, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0].token_count > 0


# ---------------------------------------------------------------------------
# Task 4: Large node splitting
# ---------------------------------------------------------------------------


class TestLargeNodeSplitting:
    def _make_large_class(self, method_count: int = 20) -> tuple[MockCodeNode, str]:
        """Erzeugt eine Klasse mit vielen Methoden."""
        lines = ["class BigClass:"]
        for i in range(method_count):
            lines.append("")
            lines.append(f"    def method_{i}(self):")
            lines.append(f"        x = {i} * 2")
            lines.append(f"        return x + {i}")
        source = "\n".join(lines) + "\n"
        end_line = len(lines)
        node = MockCodeNode(
            id="cls-big",
            node_type="class",
            name="BigClass",
            start_line=1,
            end_line=end_line,
        )
        return node, source

    def test_large_node_split_into_multiple_chunks(self):
        """Grosse Klasse mit 20 Methoden wird bei max_tokens=50 in mehrere Chunks aufgeteilt."""
        node, source = self._make_large_class(method_count=20)
        chunks = chunk_node(node, source, max_tokens=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.parent_node_id == node.id

    def test_large_node_chunks_cover_all_content(self):
        """Alle 100 Zeilen sind in den Chunks enthalten."""
        lines = [f"line_{i} = {i}" for i in range(100)]
        source = "\n".join(lines) + "\n"
        node = MockCodeNode(
            id="mod-1", node_type="module", name="big_mod", start_line=1, end_line=100
        )
        chunks = chunk_node(node, source, max_tokens=50)
        reconstructed = "".join(c.content for c in chunks)
        assert "line_0 = 0" in reconstructed
        assert "line_99 = 99" in reconstructed

    def test_large_node_chunk_ids_sequential(self):
        """Chunk-IDs sind sequentiell: {node.id}:chunk-{i}."""
        node, source = self._make_large_class(method_count=20)
        chunks = chunk_node(node, source, max_tokens=50)
        for i, chunk in enumerate(chunks):
            assert chunk.id == f"{node.id}:chunk-{i}"

    def test_large_node_chunks_respect_max_tokens(self):
        """Kein Chunk ueberschreitet max_tokens * 2."""
        node, source = self._make_large_class(method_count=20)
        max_t = 50
        chunks = chunk_node(node, source, max_tokens=max_t)
        for chunk in chunks:
            assert chunk.token_count <= max_t * 2, (
                f"Chunk {chunk.id} hat {chunk.token_count} Tokens (max erlaubt: {max_t * 2})"
            )

    def test_large_node_line_numbers_correct(self):
        """Erster Chunk beginnt bei node.start_line, letzter endet bei node.end_line."""
        node, source = self._make_large_class(method_count=20)
        chunks = chunk_node(node, source, max_tokens=50)
        assert chunks[0].start_line == node.start_line
        assert chunks[-1].end_line == node.end_line
