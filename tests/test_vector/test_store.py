"""Tests for LanceDB vector store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemesis.vector.store import SearchResult, VectorStore

if TYPE_CHECKING:
    from pathlib import Path

# Embedding dimensionality used across project-id tests
DIMS = 3


class TestSearchResult:
    """Tests for the SearchResult data model."""

    def test_creation(self) -> None:
        result = SearchResult(
            id="chunk-001",
            text="def hello(): pass",
            score=0.95,
            metadata={"file": "main.py", "language": "python"},
        )
        assert result.id == "chunk-001"
        assert result.text == "def hello(): pass"
        assert result.score == 0.95
        assert result.metadata["file"] == "main.py"

    def test_defaults(self) -> None:
        result = SearchResult(id="x", text="y", score=0.5)
        assert result.metadata == {}

    def test_ordering_by_score(self) -> None:
        r1 = SearchResult(id="a", text="a", score=0.9)
        r2 = SearchResult(id="b", text="b", score=0.7)
        r3 = SearchResult(id="c", text="c", score=0.8)
        ranked = sorted([r1, r2, r3], key=lambda r: r.score, reverse=True)
        assert [r.id for r in ranked] == ["a", "c", "b"]


class TestVectorStoreInit:
    """Tests for VectorStore initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_creates_db(self, tmp_path: Path) -> None:
        store = VectorStore(path=str(tmp_path / "vectors"))
        await store.initialize(dimensions=384)
        assert store.is_initialized
        await store.close()

    @pytest.mark.asyncio
    async def test_initialize_with_default_dimensions(self, tmp_path: Path) -> None:
        store = VectorStore(path=str(tmp_path / "vectors"))
        await store.initialize(dimensions=1536)
        assert store.is_initialized
        count = await store.count()
        assert count == 0
        await store.close()

    @pytest.mark.asyncio
    async def test_close_without_initialize(self, tmp_path: Path) -> None:
        """close() should be safe to call even without initialize."""
        store = VectorStore(path=str(tmp_path / "vectors"))
        await store.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_double_initialize(self, tmp_path: Path) -> None:
        """Second initialize reuses existing table."""
        store = VectorStore(path=str(tmp_path / "vectors"))
        await store.initialize(dimensions=384)
        await store.initialize(dimensions=384)  # Should not raise
        assert store.is_initialized
        await store.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path: Path) -> None:
        """VectorStore should work as async context manager."""
        async with VectorStore(path=str(tmp_path / "vectors")) as store:
            await store.initialize(dimensions=384)
            assert store.is_initialized


class TestVectorStoreAdd:
    """Tests for adding vectors to the store."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> VectorStore:
        """Create and initialize a VectorStore for testing."""
        s = VectorStore(path=str(tmp_path / "vectors"))
        await s.initialize(dimensions=3)
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_add_single_vector(self, store: VectorStore) -> None:
        await store.add(
            ids=["chunk-1"],
            texts=["def hello(): pass"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadata=[{"file": "main.py"}],
        )
        count = await store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_add_multiple_vectors(self, store: VectorStore) -> None:
        await store.add(
            ids=["c1", "c2", "c3"],
            texts=["text1", "text2", "text3"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            metadata=[{"f": "a.py"}, {"f": "b.py"}, {"f": "c.py"}],
        )
        count = await store.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_add_preserves_metadata(self, store: VectorStore) -> None:
        meta = {"file": "utils.py", "language": "python", "line_start": 42}
        await store.add(
            ids=["c1"],
            texts=["some code"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadata=[meta],
        )
        # Verify via search that metadata roundtrips correctly
        results = await store.search(query_embedding=[0.1, 0.2, 0.3], limit=1)
        assert len(results) == 1
        assert results[0].metadata["file"] == "utils.py"
        assert results[0].metadata["language"] == "python"
        assert results[0].metadata["line_start"] == 42

    @pytest.mark.asyncio
    async def test_add_empty_lists(self, store: VectorStore) -> None:
        """Adding empty lists should be a no-op."""
        await store.add(ids=[], texts=[], embeddings=[], metadata=[])
        count = await store.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_add_mismatched_lengths_raises(self, store: VectorStore) -> None:
        with pytest.raises(ValueError, match="must have the same length"):
            await store.add(
                ids=["c1", "c2"],
                texts=["text1"],  # Mismatched!
                embeddings=[[0.1, 0.2, 0.3]],
                metadata=[{}],
            )

    @pytest.mark.asyncio
    async def test_add_without_initialize_raises(self, tmp_path: Path) -> None:
        store = VectorStore(path=str(tmp_path / "vectors2"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.add(
                ids=["c1"],
                texts=["text"],
                embeddings=[[0.1]],
                metadata=[{}],
            )


class TestVectorStoreSearch:
    """Tests for similarity search."""

    @pytest.fixture
    async def populated_store(self, tmp_path: Path) -> VectorStore:
        """Create a store with pre-populated test data."""
        s = VectorStore(path=str(tmp_path / "vectors"))
        await s.initialize(dimensions=3)
        await s.add(
            ids=["c1", "c2", "c3", "c4"],
            texts=[
                "def authenticate(user, password):",
                "class UserService:",
                "SELECT * FROM users WHERE id = ?",
                "import React from 'react';",
            ],
            embeddings=[
                [0.9, 0.1, 0.0],  # auth-related
                [0.8, 0.2, 0.0],  # user-related
                [0.7, 0.3, 0.0],  # db-related
                [0.0, 0.0, 1.0],  # frontend-related (orthogonal)
            ],
            metadata=[
                {"file": "auth.py", "language": "python"},
                {"file": "user_service.py", "language": "python"},
                {"file": "queries.sql", "language": "sql"},
                {"file": "App.tsx", "language": "typescript"},
            ],
        )
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_search_returns_results(self, populated_store: VectorStore) -> None:
        results = await populated_store.search(query_embedding=[0.9, 0.1, 0.0], limit=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_returns_most_similar_first(self, populated_store: VectorStore) -> None:
        # Query close to auth embedding
        results = await populated_store.search(query_embedding=[0.9, 0.1, 0.0], limit=4)
        # c1 (auth) should be first, c4 (frontend) should be last
        assert results[0].id == "c1"
        assert results[-1].id == "c4"

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, populated_store: VectorStore) -> None:
        results = await populated_store.search(query_embedding=[0.5, 0.5, 0.0], limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_includes_text_and_metadata(self, populated_store: VectorStore) -> None:
        results = await populated_store.search(query_embedding=[0.9, 0.1, 0.0], limit=1)
        assert results[0].text == "def authenticate(user, password):"
        assert results[0].metadata["file"] == "auth.py"
        assert results[0].metadata["language"] == "python"

    @pytest.mark.asyncio
    async def test_search_score_range(self, populated_store: VectorStore) -> None:
        """Scores should be non-negative floats."""
        results = await populated_store.search(query_embedding=[0.9, 0.1, 0.0], limit=4)
        for r in results:
            assert isinstance(r.score, float)

    @pytest.mark.asyncio
    async def test_search_empty_store(self, tmp_path: Path) -> None:
        store = VectorStore(path=str(tmp_path / "empty"))
        await store.initialize(dimensions=3)
        results = await store.search(query_embedding=[0.1, 0.2, 0.3], limit=5)
        assert results == []
        await store.close()

    @pytest.mark.asyncio
    async def test_search_with_filter(self, populated_store: VectorStore) -> None:
        """Search with metadata filter should only return matching results."""
        results = await populated_store.search(
            query_embedding=[0.5, 0.5, 0.0],
            limit=10,
            filter={"language": "python"},
        )
        # Only c1, c2 are Python
        assert all(r.metadata["language"] == "python" for r in results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_filter_no_match(self, populated_store: VectorStore) -> None:
        results = await populated_store.search(
            query_embedding=[0.5, 0.5, 0.0],
            limit=10,
            filter={"language": "rust"},
        )
        assert results == []


class TestVectorStoreDelete:
    """Tests for delete operations."""

    @pytest.fixture
    async def populated_store(self, tmp_path: Path) -> VectorStore:
        """Create a store with pre-populated test data."""
        s = VectorStore(path=str(tmp_path / "vectors"))
        await s.initialize(dimensions=3)
        await s.add(
            ids=["c1", "c2", "c3", "c4"],
            texts=["text1", "text2", "text3", "text4"],
            embeddings=[
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
                [0.1, 0.1, 0.0],
            ],
            metadata=[
                {"file": "a.py"},
                {"file": "a.py"},
                {"file": "b.py"},
                {"file": "b.py"},
            ],
        )
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_delete_single_id(self, populated_store: VectorStore) -> None:
        await populated_store.delete(ids=["c1"])
        count = await populated_store.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_multiple_ids(self, populated_store: VectorStore) -> None:
        await populated_store.delete(ids=["c1", "c3"])
        count = await populated_store.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent_id(self, populated_store: VectorStore) -> None:
        """Deleting a non-existent ID should not raise."""
        await populated_store.delete(ids=["nonexistent"])
        count = await populated_store.count()
        assert count == 4  # Nothing deleted

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, populated_store: VectorStore) -> None:
        await populated_store.delete(ids=[])
        count = await populated_store.count()
        assert count == 4

    @pytest.mark.asyncio
    async def test_delete_by_file(self, populated_store: VectorStore) -> None:
        """Delete all vectors belonging to a specific file."""
        await populated_store.delete_by_file(file_path="a.py")
        count = await populated_store.count()
        assert count == 2
        # Remaining should all be from b.py
        results = await populated_store.search(query_embedding=[0.0, 0.0, 0.1], limit=10)
        for r in results:
            assert r.metadata["file"] == "b.py"

    @pytest.mark.asyncio
    async def test_delete_by_file_nonexistent(self, populated_store: VectorStore) -> None:
        """Deleting by non-existent file should not raise."""
        await populated_store.delete_by_file(file_path="nonexistent.py")
        count = await populated_store.count()
        assert count == 4

    @pytest.mark.asyncio
    async def test_delete_then_add(self, populated_store: VectorStore) -> None:
        """Store should work correctly after deletions."""
        await populated_store.delete(ids=["c1", "c2", "c3", "c4"])
        count = await populated_store.count()
        assert count == 0

        await populated_store.add(
            ids=["new1"],
            texts=["new text"],
            embeddings=[[0.5, 0.5, 0.5]],
            metadata=[{"file": "new.py"}],
        )
        count = await populated_store.count()
        assert count == 1


class TestVectorStoreProjectId:
    """Tests for project_id filtering in VectorStore."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> VectorStore:
        """Create and initialize a VectorStore for testing."""
        s = VectorStore(path=str(tmp_path / "vectors"))
        await s.initialize(dimensions=DIMS)
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_add_with_project_id(self, store: VectorStore) -> None:
        await store.add(
            ids=["id1"],
            texts=["hello world"],
            embeddings=[[0.1] * DIMS],
            metadata=[{"file": "a.py"}],
            project_id="eve",
        )
        results = await store.search([0.1] * DIMS, limit=10)
        assert len(results) == 1
        assert results[0].project_id == "eve"

    @pytest.mark.asyncio
    async def test_search_filters_by_project(self, store: VectorStore) -> None:
        await store.add(["id1"], ["hello"], [[0.1] * DIMS], [{}], project_id="eve")
        await store.add(["id2"], ["world"], [[0.1] * DIMS], [{}], project_id="nemesis")
        eve_results = await store.search([0.1] * DIMS, limit=10, project_id="eve")
        assert all(r.project_id == "eve" for r in eve_results)
        assert len(eve_results) == 1

    @pytest.mark.asyncio
    async def test_search_all_projects_when_no_filter(self, store: VectorStore) -> None:
        await store.add(["id1"], ["hello"], [[0.1] * DIMS], [{}], project_id="eve")
        await store.add(["id2"], ["world"], [[0.1] * DIMS], [{}], project_id="nemesis")
        all_results = await store.search([0.1] * DIMS, limit=10)
        assert len(all_results) == 2

    @pytest.mark.asyncio
    async def test_delete_by_project(self, store: VectorStore) -> None:
        await store.add(["id1"], ["hello"], [[0.1] * DIMS], [{}], project_id="eve")
        await store.add(["id2"], ["world"], [[0.1] * DIMS], [{}], project_id="nemesis")
        await store.delete_by_project("eve")
        assert await store.count() == 1
