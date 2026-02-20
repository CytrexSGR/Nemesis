# 04b — Vector Store Core

> **Arbeitspaket D2** — Teil 2 von 3 des Vector Store Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** VectorStore mit LanceDB implementieren: Initialisierung, Hinzufuegen und Similarity Search. Umfasst Tasks 4, 5, 6 des urspruenglichen Vector-Store-Plans.

**Architecture:** `VectorStore` wraps LanceDB fuer add/search Operationen. Alle Operationen sind async. Tests nutzen temporaere LanceDB-Verzeichnisse.

**Tech Stack:** LanceDB, PyArrow, numpy

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [01-project-setup](01-project-setup.md), [04a-embedding-providers](04a-embedding-providers.md)

**Navigation:** [← Vorheriges Paket: D1 — Embedding Providers](04a-embedding-providers.md) | [Nächstes Paket: D3 — Vector Store Advanced →](04c-vector-store-advanced.md)

---

## Task 4: SearchResult Model & VectorStore Initialization

**Files:**
- `tests/test_vector/test_store.py`
- `nemesis/vector/store.py`

### Step 1 — Write failing test

```python
# tests/test_vector/test_store.py
"""Tests for LanceDB vector store."""
from __future__ import annotations

import pytest
from pathlib import Path

from nemesis.vector.store import SearchResult, VectorStore


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
    async def test_initialize_with_default_dimensions(
        self, tmp_path: Path
    ) -> None:
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
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_store.py -x -v 2>&1 | head -20
```

### Step 3 — Implement

```python
# nemesis/vector/store.py
"""LanceDB-based vector store for Nemesis.

Stores code chunk embeddings and supports similarity search
with optional metadata filtering.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa


@dataclass(frozen=True)
class SearchResult:
    """A single result from a vector similarity search.

    Attributes:
        id: Unique identifier of the stored vector.
        text: Original text content of the chunk.
        score: Similarity score (higher = more similar).
        metadata: Additional metadata stored with the vector.
    """

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# LanceDB table name
_TABLE_NAME = "chunks"


class VectorStore:
    """LanceDB vector store for code chunk embeddings.

    Provides async add/search/delete operations backed by LanceDB.
    Uses a single table with a fixed schema determined at initialization.

    Args:
        path: Directory path for LanceDB storage.
        table_name: Name of the LanceDB table.
    """

    def __init__(
        self,
        path: str,
        table_name: str = _TABLE_NAME,
    ) -> None:
        self._path = path
        self._table_name = table_name
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._dimensions: int | None = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the store has been initialized."""
        return self._initialized

    async def initialize(self, dimensions: int) -> None:
        """Initialize the vector store with the given embedding dimensions.

        Creates the LanceDB database and table if they don't exist.
        If the table already exists, it is reused.

        Args:
            dimensions: Dimensionality of the embedding vectors.
        """
        self._dimensions = dimensions

        loop = asyncio.get_event_loop()
        self._db = await loop.run_in_executor(
            None, lambda: lancedb.connect(self._path)
        )

        schema = pa.schema(
            [
                pa.field("id", pa.utf8()),
                pa.field("text", pa.utf8()),
                pa.field(
                    "vector", pa.list_(pa.float32(), list_size=dimensions)
                ),
                pa.field("metadata", pa.utf8()),  # JSON-encoded
            ]
        )

        existing_tables = await loop.run_in_executor(
            None, lambda: self._db.table_names()
        )

        if self._table_name in existing_tables:
            self._table = await loop.run_in_executor(
                None, lambda: self._db.open_table(self._table_name)
            )
        else:
            self._table = await loop.run_in_executor(
                None,
                lambda: self._db.create_table(
                    self._table_name, schema=schema
                ),
            )

        self._initialized = True

    async def count(self) -> int:
        """Return the number of vectors in the store.

        Returns:
            The total number of stored vectors.
        """
        if not self._initialized or self._table is None:
            return 0

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._table.count_rows()
        )
        return result

    async def close(self) -> None:
        """Close the vector store and release resources."""
        self._table = None
        self._db = None
        self._initialized = False

    async def __aenter__(self) -> VectorStore:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit — closes the store."""
        await self.close()
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_store.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/vector/store.py tests/test_vector/test_store.py
git commit -m "feat(vector): add VectorStore init, SearchResult model, LanceDB setup

TDD Task 4/8 of 04-vector-store plan.
SearchResult dataclass for query results. VectorStore wraps LanceDB
with async init, schema creation, context manager, and count."
```

---

## Task 5: VectorStore — Add Vectors

**Files:**
- `tests/test_vector/test_store.py` (extend)
- `nemesis/vector/store.py` (extend)

### Step 1 — Write failing test

```python
# tests/test_vector/test_store.py — APPEND to existing file
import json


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
        results = await store.search(
            query_embedding=[0.1, 0.2, 0.3], limit=1
        )
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
    async def test_add_mismatched_lengths_raises(
        self, store: VectorStore
    ) -> None:
        with pytest.raises(ValueError, match="must have the same length"):
            await store.add(
                ids=["c1", "c2"],
                texts=["text1"],  # Mismatched!
                embeddings=[[0.1, 0.2, 0.3]],
                metadata=[{}],
            )

    @pytest.mark.asyncio
    async def test_add_without_initialize_raises(
        self, tmp_path: Path
    ) -> None:
        store = VectorStore(path=str(tmp_path / "vectors2"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.add(
                ids=["c1"],
                texts=["text"],
                embeddings=[[0.1]],
                metadata=[{}],
            )
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_store.py::TestVectorStoreAdd -x -v 2>&1 | head -20
```

### Step 3 — Implement

Add the `add` method to `VectorStore` in `nemesis/vector/store.py`:

```python
    # Add this method to the VectorStore class

    def _require_initialized(self) -> None:
        """Raise RuntimeError if the store is not initialized."""
        if not self._initialized or self._table is None:
            raise RuntimeError(
                "VectorStore is not initialized. Call initialize() first."
            )

    async def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add vectors with their texts and metadata to the store.

        Args:
            ids: Unique identifiers for each vector.
            texts: Original text content for each vector.
            embeddings: Embedding vectors.
            metadata: Metadata dicts for each vector.

        Raises:
            ValueError: If input lists have different lengths.
            RuntimeError: If the store is not initialized.
        """
        self._require_initialized()

        lengths = {len(ids), len(texts), len(embeddings), len(metadata)}
        if len(lengths) > 1:
            raise ValueError(
                "ids, texts, embeddings, and metadata must have the same length. "
                f"Got lengths: ids={len(ids)}, texts={len(texts)}, "
                f"embeddings={len(embeddings)}, metadata={len(metadata)}"
            )

        if not ids:
            return

        import json

        rows = [
            {
                "id": id_,
                "text": text,
                "vector": embedding,
                "metadata": json.dumps(metadata_item),
            }
            for id_, text, embedding, metadata_item in zip(
                ids, texts, embeddings, metadata
            )
        ]

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._table.add(rows))
```

Also update the `count()` method to use `_require_initialized` pattern (but keep it safe for uninitialized):

The `count` method already handles the uninitialized case gracefully. No change needed there.

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_store.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/vector/store.py tests/test_vector/test_store.py
git commit -m "feat(vector): implement VectorStore.add() with validation

TDD Task 5/8 of 04-vector-store plan.
Add vectors with texts and JSON-serialized metadata to LanceDB.
Input validation ensures all lists have matching lengths.
Guards against calling add() before initialize()."
```

---

## Task 6: VectorStore — Similarity Search

**Files:**
- `tests/test_vector/test_store.py` (extend)
- `nemesis/vector/store.py` (extend)

### Step 1 — Write failing test

```python
# tests/test_vector/test_store.py — APPEND to existing file


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
    async def test_search_returns_results(
        self, populated_store: VectorStore
    ) -> None:
        results = await populated_store.search(
            query_embedding=[0.9, 0.1, 0.0], limit=2
        )
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_returns_most_similar_first(
        self, populated_store: VectorStore
    ) -> None:
        # Query close to auth embedding
        results = await populated_store.search(
            query_embedding=[0.9, 0.1, 0.0], limit=4
        )
        # c1 (auth) should be first, c4 (frontend) should be last
        assert results[0].id == "c1"
        assert results[-1].id == "c4"

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self, populated_store: VectorStore
    ) -> None:
        results = await populated_store.search(
            query_embedding=[0.5, 0.5, 0.0], limit=2
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_includes_text_and_metadata(
        self, populated_store: VectorStore
    ) -> None:
        results = await populated_store.search(
            query_embedding=[0.9, 0.1, 0.0], limit=1
        )
        assert results[0].text == "def authenticate(user, password):"
        assert results[0].metadata["file"] == "auth.py"
        assert results[0].metadata["language"] == "python"

    @pytest.mark.asyncio
    async def test_search_score_range(
        self, populated_store: VectorStore
    ) -> None:
        """Scores should be non-negative floats."""
        results = await populated_store.search(
            query_embedding=[0.9, 0.1, 0.0], limit=4
        )
        for r in results:
            assert isinstance(r.score, float)

    @pytest.mark.asyncio
    async def test_search_empty_store(self, tmp_path: Path) -> None:
        store = VectorStore(path=str(tmp_path / "empty"))
        await store.initialize(dimensions=3)
        results = await store.search(
            query_embedding=[0.1, 0.2, 0.3], limit=5
        )
        assert results == []
        await store.close()

    @pytest.mark.asyncio
    async def test_search_with_filter(
        self, populated_store: VectorStore
    ) -> None:
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
    async def test_search_filter_no_match(
        self, populated_store: VectorStore
    ) -> None:
        results = await populated_store.search(
            query_embedding=[0.5, 0.5, 0.0],
            limit=10,
            filter={"language": "rust"},
        )
        assert results == []
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_store.py::TestVectorStoreSearch -x -v 2>&1 | head -20
```

### Step 3 — Implement

Add the `search` method to `VectorStore` in `nemesis/vector/store.py`:

```python
    # Add this method to the VectorStore class

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for the most similar vectors.

        Args:
            query_embedding: The query vector to search against.
            limit: Maximum number of results to return.
            filter: Optional metadata filter. Keys are metadata field names,
                    values are exact-match values. Filters are applied as
                    SQL WHERE clauses on the JSON metadata column.

        Returns:
            List of SearchResult ordered by similarity (best first).
        """
        self._require_initialized()

        import json

        loop = asyncio.get_event_loop()

        row_count = await self.count()
        if row_count == 0:
            return []

        def _search() -> list[dict]:
            query = self._table.search(query_embedding).limit(limit)

            if filter:
                # Build a WHERE clause that filters on the JSON metadata
                # LanceDB stores metadata as a utf8 column, so we use
                # JSON extraction in the SQL filter.
                conditions = []
                for key, value in filter.items():
                    escaped_value = json.dumps(value)
                    conditions.append(
                        f"json_extract_string(metadata, '$.{key}') = {escaped_value}"
                    )
                where_clause = " AND ".join(conditions)
                query = query.where(where_clause)

            return query.to_list()

        raw_results = await loop.run_in_executor(None, _search)

        results = []
        for row in raw_results:
            meta = json.loads(row["metadata"]) if row.get("metadata") else {}
            results.append(
                SearchResult(
                    id=row["id"],
                    text=row["text"],
                    score=float(1.0 - row.get("_distance", 0.0)),
                    metadata=meta,
                )
            )

        return results
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_store.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/vector/store.py tests/test_vector/test_store.py
git commit -m "feat(vector): implement VectorStore.search() with metadata filtering

TDD Task 6/8 of 04-vector-store plan.
Similarity search via LanceDB with configurable limit.
Supports metadata filtering via JSON extraction in WHERE clauses.
Returns SearchResult list sorted by similarity score."
```

---

## Summary

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 4 | SearchResult Model + VectorStore Init | `store.py` | 8 |
| 5 | VectorStore.add() mit Validierung | `store.py` | 6 |
| 6 | VectorStore.search() mit Metadata-Filter | `store.py` | 8 |
| **Gesamt** | | **1 Datei** | **22 Tests** |

### Dependencies (pyproject.toml additions)

```toml
[project]
dependencies = [
    # ... existing ...
    "lancedb>=0.15",
    "pyarrow>=14.0",
]
```

---

**Navigation:** [← Vorheriges Paket: D1 — Embedding Providers](04a-embedding-providers.md) | [Nächstes Paket: D3 — Vector Store Advanced →](04c-vector-store-advanced.md)
