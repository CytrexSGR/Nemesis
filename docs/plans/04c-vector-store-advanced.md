# 04c — Vector Store Advanced

> **Arbeitspaket D3** — Teil 3 von 3 des Vector Store Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete-Operationen und Pipeline-Helpers (Factory-Funktion + Integrationstests) fuer den VectorStore. Umfasst Tasks 7, 8 des urspruenglichen Vector-Store-Plans.

**Architecture:** `VectorStore.delete()` und `delete_by_file()` fuer inkrementelles Re-Indexing. `create_embedding_provider()` Factory-Funktion im `__init__.py`. Integrationstests pruefen den vollstaendigen embed-store-search Roundtrip.

**Tech Stack:** LanceDB, OpenAI API (gemockt), PyArrow

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [01-project-setup](01-project-setup.md), [04a-embedding-providers](04a-embedding-providers.md), [04b-vector-store-core](04b-vector-store-core.md)

**Navigation:** [← Vorheriges Paket: D2 — Vector Store Core](04b-vector-store-core.md)

---

## Task 7: VectorStore — Delete Operations

**Files:**
- `tests/test_vector/test_store.py` (extend)
- `nemesis/vector/store.py` (extend)

### Step 1 — Write failing test

```python
# tests/test_vector/test_store.py — APPEND to existing file


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
    async def test_delete_single_id(
        self, populated_store: VectorStore
    ) -> None:
        await populated_store.delete(ids=["c1"])
        count = await populated_store.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_multiple_ids(
        self, populated_store: VectorStore
    ) -> None:
        await populated_store.delete(ids=["c1", "c3"])
        count = await populated_store.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent_id(
        self, populated_store: VectorStore
    ) -> None:
        """Deleting a non-existent ID should not raise."""
        await populated_store.delete(ids=["nonexistent"])
        count = await populated_store.count()
        assert count == 4  # Nothing deleted

    @pytest.mark.asyncio
    async def test_delete_empty_list(
        self, populated_store: VectorStore
    ) -> None:
        await populated_store.delete(ids=[])
        count = await populated_store.count()
        assert count == 4

    @pytest.mark.asyncio
    async def test_delete_by_file(
        self, populated_store: VectorStore
    ) -> None:
        """Delete all vectors belonging to a specific file."""
        await populated_store.delete_by_file(file_path="a.py")
        count = await populated_store.count()
        assert count == 2
        # Remaining should all be from b.py
        results = await populated_store.search(
            query_embedding=[0.0, 0.0, 0.1], limit=10
        )
        for r in results:
            assert r.metadata["file"] == "b.py"

    @pytest.mark.asyncio
    async def test_delete_by_file_nonexistent(
        self, populated_store: VectorStore
    ) -> None:
        """Deleting by non-existent file should not raise."""
        await populated_store.delete_by_file(file_path="nonexistent.py")
        count = await populated_store.count()
        assert count == 4

    @pytest.mark.asyncio
    async def test_delete_then_add(
        self, populated_store: VectorStore
    ) -> None:
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
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_store.py::TestVectorStoreDelete -x -v 2>&1 | head -20
```

### Step 3 — Implement

Add `delete` and `delete_by_file` methods to `VectorStore` in `nemesis/vector/store.py`:

```python
    # Add these methods to the VectorStore class

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete.
        """
        self._require_initialized()

        if not ids:
            return

        loop = asyncio.get_event_loop()

        # Build an IN clause for the delete filter
        id_list = ", ".join(f"'{id_}'" for id_ in ids)
        where = f"id IN ({id_list})"

        await loop.run_in_executor(
            None, lambda: self._table.delete(where)
        )

    async def delete_by_file(self, file_path: str) -> None:
        """Delete all vectors associated with a specific file.

        Uses JSON extraction on the metadata column to match the
        file path.

        Args:
            file_path: The file path to match in metadata.
        """
        self._require_initialized()

        import json

        loop = asyncio.get_event_loop()
        escaped = json.dumps(file_path)
        where = f"json_extract_string(metadata, '$.file') = {escaped}"

        await loop.run_in_executor(
            None, lambda: self._table.delete(where)
        )
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_store.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/vector/store.py tests/test_vector/test_store.py
git commit -m "feat(vector): implement VectorStore.delete() and delete_by_file()

TDD Task 7/8 of 04-vector-store plan.
Delete by ID list or by file path (via JSON metadata extraction).
Supports incremental re-indexing: delete old file chunks, add new ones."
```

---

## Task 8: Integration — Embed & Store Pipeline Helpers

**Files:**
- `tests/test_vector/test_integration.py`
- `nemesis/vector/__init__.py` (extend with convenience factory)

### Step 1 — Write failing test

```python
# tests/test_vector/test_integration.py
"""Integration tests for the vector module.

Tests the full flow: embed texts → store in LanceDB → search.
Uses mocked OpenAI client + real (temporary) LanceDB.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nemesis.vector.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    OpenAIEmbeddings,
)
from nemesis.vector.store import SearchResult, VectorStore
from nemesis.vector import create_embedding_provider


class TestCreateEmbeddingProvider:
    """Tests for the factory function."""

    def test_create_openai_provider(self) -> None:
        mock_client = MagicMock()
        provider = create_embedding_provider(
            provider_type="openai", client=mock_client
        )
        assert isinstance(provider, OpenAIEmbeddings)
        assert provider.dimensions == 1536

    def test_create_openai_provider_custom_model(self) -> None:
        mock_client = MagicMock()
        provider = create_embedding_provider(
            provider_type="openai",
            client=mock_client,
            model="text-embedding-3-large",
            dimensions=3072,
        )
        assert provider.dimensions == 3072
        assert provider.model_name == "text-embedding-3-large"

    def test_create_local_provider(self) -> None:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        provider = create_embedding_provider(
            provider_type="local", model=mock_model
        )
        assert provider.dimensions == 384

    def test_create_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_embedding_provider(provider_type="unknown")


class TestEmbedAndStore:
    """End-to-end: embed → store → search with mocked embeddings."""

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock()
        return client

    @pytest.fixture
    def mock_provider(self, mock_openai_client: MagicMock) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            client=mock_openai_client, dimensions=4
        )

    def _make_mock_response(
        self, embeddings: list[list[float]], total_tokens: int = 10
    ) -> MagicMock:
        mock_data = []
        for i, emb in enumerate(embeddings):
            item = MagicMock()
            item.embedding = emb
            item.index = i
            mock_data.append(item)

        mock_usage = MagicMock()
        mock_usage.total_tokens = total_tokens

        mock_response = MagicMock()
        mock_response.data = mock_data
        mock_response.usage = mock_usage
        return mock_response

    @pytest.mark.asyncio
    async def test_embed_store_search_roundtrip(
        self,
        tmp_path: Path,
        mock_provider: OpenAIEmbeddings,
        mock_openai_client: MagicMock,
    ) -> None:
        """Full pipeline: embed code chunks, store them, search."""
        # --- Setup mock embeddings ---
        chunk_embeddings = [
            [1.0, 0.0, 0.0, 0.0],  # auth
            [0.0, 1.0, 0.0, 0.0],  # database
            [0.0, 0.0, 1.0, 0.0],  # frontend
        ]
        mock_openai_client.embeddings.create.return_value = (
            self._make_mock_response(chunk_embeddings, total_tokens=30)
        )

        # --- Embed ---
        texts = [
            "def authenticate(user, password): ...",
            "SELECT * FROM users WHERE active = true",
            "const App = () => <div>Hello</div>",
        ]
        result = await mock_provider.embed(texts)
        assert len(result.embeddings) == 3

        # --- Store ---
        store = VectorStore(path=str(tmp_path / "vectors"))
        await store.initialize(dimensions=4)
        await store.add(
            ids=["chunk-auth", "chunk-db", "chunk-ui"],
            texts=texts,
            embeddings=result.embeddings,
            metadata=[
                {"file": "auth.py", "type": "function"},
                {"file": "queries.sql", "type": "query"},
                {"file": "App.tsx", "type": "component"},
            ],
        )
        assert await store.count() == 3

        # --- Search (auth-like query) ---
        query_embedding = [0.9, 0.1, 0.0, 0.0]
        results = await store.search(query_embedding=query_embedding, limit=2)
        assert len(results) == 2
        assert results[0].id == "chunk-auth"
        assert results[0].metadata["file"] == "auth.py"

        # --- Search with filter ---
        results = await store.search(
            query_embedding=[0.5, 0.5, 0.0, 0.0],
            limit=10,
            filter={"type": "function"},
        )
        assert len(results) == 1
        assert results[0].id == "chunk-auth"

        await store.close()

    @pytest.mark.asyncio
    async def test_incremental_update_flow(
        self,
        tmp_path: Path,
        mock_provider: OpenAIEmbeddings,
        mock_openai_client: MagicMock,
    ) -> None:
        """Simulate delta update: delete old chunks for a file, add new ones."""
        store = VectorStore(path=str(tmp_path / "vectors"))
        await store.initialize(dimensions=4)

        # --- Initial indexing ---
        mock_openai_client.embeddings.create.return_value = (
            self._make_mock_response(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                total_tokens=20,
            )
        )
        initial_texts = ["old function v1", "old class v1"]
        result = await mock_provider.embed(initial_texts)

        await store.add(
            ids=["f1-chunk1", "f1-chunk2"],
            texts=initial_texts,
            embeddings=result.embeddings,
            metadata=[
                {"file": "service.py", "version": 1},
                {"file": "service.py", "version": 1},
            ],
        )
        assert await store.count() == 2

        # --- File changed: delete old, add new ---
        await store.delete_by_file(file_path="service.py")
        assert await store.count() == 0

        mock_openai_client.embeddings.create.return_value = (
            self._make_mock_response(
                [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5]],
                total_tokens=30,
            )
        )
        new_texts = ["new function v2", "new class v2", "new method v2"]
        result = await mock_provider.embed(new_texts)

        await store.add(
            ids=["f1-chunk1-v2", "f1-chunk2-v2", "f1-chunk3-v2"],
            texts=new_texts,
            embeddings=result.embeddings,
            metadata=[
                {"file": "service.py", "version": 2},
                {"file": "service.py", "version": 2},
                {"file": "service.py", "version": 2},
            ],
        )
        assert await store.count() == 3

        # --- Verify new data is searchable ---
        results = await store.search(
            query_embedding=[0.5, 0.5, 0.0, 0.0], limit=1
        )
        assert results[0].metadata["version"] == 2

        await store.close()
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_integration.py -x -v 2>&1 | head -20
```

Tests fail because `create_embedding_provider` does not exist yet.

### Step 3 — Implement

```python
# nemesis/vector/__init__.py — REPLACE entire file
"""Vector storage and embedding generation for Nemesis.

This module provides:
- EmbeddingProvider protocol and implementations (OpenAI, local)
- VectorStore backed by LanceDB for similarity search
- Factory function for creating embedding providers
"""
from __future__ import annotations

from typing import Any

from nemesis.vector.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    LocalEmbeddings,
    OpenAIEmbeddings,
)
from nemesis.vector.store import SearchResult, VectorStore


def create_embedding_provider(
    provider_type: str = "openai",
    **kwargs: Any,
) -> EmbeddingProvider:
    """Factory function to create an embedding provider.

    Args:
        provider_type: Either "openai" or "local".
        **kwargs: Passed to the provider constructor.
            For "openai": client (required), model, dimensions, max_batch_size.
            For "local": model, model_name.

    Returns:
        An EmbeddingProvider instance.

    Raises:
        ValueError: If provider_type is not recognized.
    """
    if provider_type == "openai":
        return OpenAIEmbeddings(**kwargs)
    elif provider_type == "local":
        return LocalEmbeddings(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type!r}. "
            f"Supported: 'openai', 'local'"
        )


__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "LocalEmbeddings",
    "OpenAIEmbeddings",
    "SearchResult",
    "VectorStore",
    "create_embedding_provider",
]
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/ -x -v
```

### Step 5 — Commit

```bash
git add nemesis/vector/__init__.py tests/test_vector/test_integration.py
git commit -m "feat(vector): add factory function and integration tests

TDD Task 8/8 of 04-vector-store plan.
create_embedding_provider() factory for provider instantiation.
Integration tests verify full embed→store→search pipeline and
incremental update (delete old chunks, add new) with mocked
OpenAI and real temporary LanceDB."
```

---

## Summary

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 7 | VectorStore.delete() + delete_by_file() | `store.py` | 7 |
| 8 | Factory-Funktion + Integrationstests | `__init__.py` | 6 |
| **Gesamt** | | **2 Dateien** | **13 Tests** |

### Dependencies (pyproject.toml additions)

```toml
[project]
dependencies = [
    # ... existing ...
    "lancedb>=0.15",
    "pyarrow>=14.0",
    "openai>=1.0",
]

[project.optional-dependencies]
local-embeddings = [
    "sentence-transformers>=2.2",
    "torch>=2.0",
]
dev = [
    # ... existing ...
    "pytest-asyncio>=0.23",
    "numpy>=1.24",  # for test mocks
]
```

---

**Navigation:** [← Vorheriges Paket: D2 — Vector Store Core](04b-vector-store-core.md)
