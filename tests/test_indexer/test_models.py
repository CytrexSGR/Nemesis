"""Tests fuer nemesis.indexer.models."""

from pathlib import Path

from nemesis.indexer.models import ChangeType, Chunk, FileChange, IndexResult


class TestChunk:
    def test_chunk_creation(self):
        chunk = Chunk(
            id="chunk-001",
            content="def hello(): pass",
            token_count=5,
            parent_node_id="node-42",
            parent_type="function",
            start_line=1,
            end_line=1,
            file_path=Path("src/main.py"),
            embedding_id="emb-001",
        )
        assert chunk.id == "chunk-001"
        assert chunk.content == "def hello(): pass"
        assert chunk.token_count == 5
        assert chunk.parent_node_id == "node-42"
        assert chunk.parent_type == "function"
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.file_path == Path("src/main.py")
        assert chunk.embedding_id == "emb-001"

    def test_chunk_defaults(self):
        chunk = Chunk(
            id="chunk-002",
            content="x = 1",
            token_count=3,
            parent_node_id="node-1",
            parent_type="module",
            start_line=10,
            end_line=10,
            file_path=Path("lib.py"),
        )
        assert chunk.embedding_id is None


class TestFileChange:
    def test_file_change_creation(self):
        change = FileChange(
            path=Path("foo.py"),
            change_type=ChangeType.MODIFIED,
            old_hash="aaa",
            new_hash="bbb",
        )
        assert change.path == Path("foo.py")
        assert change.change_type == ChangeType.MODIFIED
        assert change.old_hash == "aaa"
        assert change.new_hash == "bbb"

    def test_file_change_added_no_old_hash(self):
        change = FileChange(
            path=Path("new.py"),
            change_type=ChangeType.ADDED,
            old_hash=None,
            new_hash="ccc",
        )
        assert change.old_hash is None
        assert change.new_hash == "ccc"

    def test_file_change_deleted_no_new_hash(self):
        change = FileChange(
            path=Path("old.py"),
            change_type=ChangeType.DELETED,
            old_hash="ddd",
            new_hash=None,
        )
        assert change.old_hash == "ddd"
        assert change.new_hash is None


class TestIndexResult:
    def test_index_result_creation(self):
        result = IndexResult(
            files_indexed=10,
            nodes_created=50,
            edges_created=30,
            chunks_created=100,
            embeddings_created=100,
            duration_ms=1234.5,
        )
        assert result.files_indexed == 10
        assert result.success is True
        assert result.errors == []

    def test_index_result_with_errors(self):
        result = IndexResult(
            files_indexed=5,
            nodes_created=20,
            edges_created=10,
            chunks_created=40,
            embeddings_created=35,
            duration_ms=999.0,
            errors=["parse error in foo.py"],
        )
        assert result.success is False
        assert len(result.errors) == 1


class TestChangeType:
    def test_change_type_enum(self):
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"
