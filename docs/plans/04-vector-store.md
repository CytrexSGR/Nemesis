# 04 — Vector Store

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the vector storage layer for Nemesis. This includes embedding generation (OpenAI + local fallback) and LanceDB-based vector search. Code chunks are embedded and stored so that natural language queries can find semantically relevant code.

**Architecture:** `EmbeddingProvider` protocol with two implementations (OpenAI, local sentence-transformers). `VectorStore` wraps LanceDB for add/search/delete operations. All operations are async. Tests use mocked OpenAI client and temporary LanceDB directories.

**Tech Stack:** LanceDB, OpenAI API (text-embedding-3-small), sentence-transformers (all-MiniLM-L6-v2), PyArrow, numpy

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [01-project-setup](01-project-setup.md)

---

## Task 1: Embedding Provider Protocol & Data Models

**Files:**
- `tests/test_vector/__init__.py`
- `tests/test_vector/test_embeddings.py`
- `nemesis/vector/__init__.py`
- `nemesis/vector/embeddings.py`

### Step 1 — Write failing test

```python
# tests/test_vector/__init__.py
```

```python
# tests/test_vector/test_embeddings.py
"""Tests for embedding provider protocol and data models."""
import pytest

from nemesis.vector.embeddings import EmbeddingProvider, EmbeddingResult


class TestEmbeddingResult:
    """Tests for the EmbeddingResult data model."""

    def test_embedding_result_creation(self) -> None:
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            total_tokens=42,
        )
        assert result.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert result.total_tokens == 42

    def test_embedding_result_defaults(self) -> None:
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]],
        )
        assert result.total_tokens == 0

    def test_embedding_result_empty(self) -> None:
        result = EmbeddingResult(embeddings=[])
        assert result.embeddings == []
        assert result.total_tokens == 0


class TestEmbeddingProviderProtocol:
    """Tests that EmbeddingProvider is a proper Protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """EmbeddingProvider must be runtime-checkable."""
        assert hasattr(EmbeddingProvider, "__protocol_attrs__") or hasattr(
            EmbeddingProvider, "__abstractmethods__"
        )

    def test_conforming_class_is_instance(self) -> None:
        """A class implementing all methods should be recognized."""

        class FakeProvider:
            async def embed(self, texts: list[str]) -> EmbeddingResult:
                return EmbeddingResult(embeddings=[[0.0] * 3])

            async def embed_single(self, text: str) -> list[float]:
                return [0.0] * 3

            @property
            def dimensions(self) -> int:
                return 3

            @property
            def model_name(self) -> str:
                return "fake"

        provider = FakeProvider()
        assert isinstance(provider, EmbeddingProvider)
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_embeddings.py -x -v 2>&1 | head -30
```

Tests fail because `nemesis/vector/embeddings.py` does not exist.

### Step 3 — Implement

```python
# nemesis/vector/__init__.py
"""Vector storage and embedding generation for Nemesis."""
```

```python
# nemesis/vector/embeddings.py
"""Embedding providers for Nemesis vector store.

Supports OpenAI text-embedding-3-small (default) and local
sentence-transformers as fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbeddingResult:
    """Result from an embedding operation.

    Attributes:
        embeddings: List of embedding vectors.
        total_tokens: Total tokens consumed by the embedding request.
    """

    embeddings: list[list[float]]
    total_tokens: int = 0


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Any class implementing these methods can be used as an embedding
    provider. Two concrete implementations ship with Nemesis:
    OpenAIEmbeddings and LocalEmbeddings.
    """

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            EmbeddingResult with one embedding vector per input text.
        """
        ...

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Convenience method — equivalent to embed([text]).embeddings[0].

        Args:
            text: The text to embed.

        Returns:
            A single embedding vector.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    @property
    def model_name(self) -> str:
        """Name/identifier of the embedding model."""
        ...
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_embeddings.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/vector/__init__.py nemesis/vector/embeddings.py \
        tests/test_vector/__init__.py tests/test_vector/test_embeddings.py
git commit -m "feat(vector): add EmbeddingProvider protocol and EmbeddingResult model

TDD Task 1/8 of 04-vector-store plan.
Defines the runtime-checkable Protocol that all embedding providers
must implement, plus the EmbeddingResult dataclass."
```

---

## Task 2: OpenAI Embeddings Provider

**Files:**
- `tests/test_vector/test_embeddings.py` (extend)
- `nemesis/vector/embeddings.py` (extend)

### Step 1 — Write failing test

```python
# tests/test_vector/test_embeddings.py — APPEND to existing file
"""Tests for OpenAI embedding provider."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nemesis.vector.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    OpenAIEmbeddings,
)


class TestOpenAIEmbeddings:
    """Tests for the OpenAI embedding provider with mocked API client."""

    def _make_mock_response(
        self, embeddings: list[list[float]], total_tokens: int = 10
    ) -> MagicMock:
        """Create a mock OpenAI embedding response."""
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

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Create a mock OpenAI async client."""
        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock()
        return client

    @pytest.fixture
    def provider(self, mock_openai_client: MagicMock) -> OpenAIEmbeddings:
        """Create an OpenAIEmbeddings with a mocked client."""
        return OpenAIEmbeddings(client=mock_openai_client)

    def test_default_properties(self, provider: OpenAIEmbeddings) -> None:
        assert provider.dimensions == 1536
        assert provider.model_name == "text-embedding-3-small"

    def test_custom_model(self, mock_openai_client: MagicMock) -> None:
        provider = OpenAIEmbeddings(
            client=mock_openai_client,
            model="text-embedding-3-large",
            dimensions=3072,
        )
        assert provider.dimensions == 3072
        assert provider.model_name == "text-embedding-3-large"

    def test_is_embedding_provider(self, provider: OpenAIEmbeddings) -> None:
        assert isinstance(provider, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_embed_single(
        self, provider: OpenAIEmbeddings, mock_openai_client: MagicMock
    ) -> None:
        mock_openai_client.embeddings.create.return_value = (
            self._make_mock_response([[0.1, 0.2, 0.3]], total_tokens=5)
        )
        result = await provider.embed_single("hello world")
        assert result == [0.1, 0.2, 0.3]
        mock_openai_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["hello world"],
            dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_embed_batch(
        self, provider: OpenAIEmbeddings, mock_openai_client: MagicMock
    ) -> None:
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_openai_client.embeddings.create.return_value = (
            self._make_mock_response(embeddings, total_tokens=15)
        )
        result = await provider.embed(["text1", "text2", "text3"])
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        assert result.total_tokens == 15
        assert result.embeddings[0] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_embed_empty_list(
        self, provider: OpenAIEmbeddings, mock_openai_client: MagicMock
    ) -> None:
        result = await provider.embed([])
        assert result.embeddings == []
        assert result.total_tokens == 0
        mock_openai_client.embeddings.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embed_batch_chunking(
        self, provider: OpenAIEmbeddings, mock_openai_client: MagicMock
    ) -> None:
        """Large batches should be split into chunks of max_batch_size."""
        provider = OpenAIEmbeddings(
            client=mock_openai_client, max_batch_size=2
        )
        # Two API calls: [text1, text2] and [text3]
        mock_openai_client.embeddings.create.side_effect = [
            self._make_mock_response([[0.1], [0.2]], total_tokens=10),
            self._make_mock_response([[0.3]], total_tokens=5),
        ]
        result = await provider.embed(["text1", "text2", "text3"])
        assert len(result.embeddings) == 3
        assert result.total_tokens == 15
        assert mock_openai_client.embeddings.create.await_count == 2

    @pytest.mark.asyncio
    async def test_embed_preserves_order(
        self, provider: OpenAIEmbeddings, mock_openai_client: MagicMock
    ) -> None:
        """Embeddings must be returned in the same order as input texts."""
        # Simulate API returning out-of-order indices
        mock_data_0 = MagicMock()
        mock_data_0.embedding = [0.9]
        mock_data_0.index = 1
        mock_data_1 = MagicMock()
        mock_data_1.embedding = [0.1]
        mock_data_1.index = 0

        mock_response = MagicMock()
        mock_response.data = [mock_data_0, mock_data_1]
        mock_response.usage = MagicMock(total_tokens=8)

        mock_openai_client.embeddings.create.return_value = mock_response

        result = await provider.embed(["first", "second"])
        # Should be reordered by index: index 0 -> [0.1], index 1 -> [0.9]
        assert result.embeddings == [[0.1], [0.9]]
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_embeddings.py::TestOpenAIEmbeddings -x -v 2>&1 | head -30
```

### Step 3 — Implement

```python
# nemesis/vector/embeddings.py — REPLACE entire file
"""Embedding providers for Nemesis vector store.

Supports OpenAI text-embedding-3-small (default) and local
sentence-transformers as fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@dataclass(frozen=True)
class EmbeddingResult:
    """Result from an embedding operation.

    Attributes:
        embeddings: List of embedding vectors.
        total_tokens: Total tokens consumed by the embedding request.
    """

    embeddings: list[list[float]]
    total_tokens: int = 0


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Any class implementing these methods can be used as an embedding
    provider. Two concrete implementations ship with Nemesis:
    OpenAIEmbeddings and LocalEmbeddings.
    """

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            EmbeddingResult with one embedding vector per input text.
        """
        ...

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A single embedding vector.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    @property
    def model_name(self) -> str:
        """Name/identifier of the embedding model."""
        ...


class OpenAIEmbeddings:
    """OpenAI embedding provider using text-embedding-3-small.

    Args:
        client: An AsyncOpenAI client instance.
        model: The OpenAI embedding model name.
        dimensions: Embedding vector dimensionality.
        max_batch_size: Maximum number of texts per API call.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        max_batch_size: int = 2048,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._max_batch_size = max_batch_size

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Name of the OpenAI embedding model."""
        return self._model

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A single embedding vector.
        """
        result = await self.embed([text])
        return result.embeddings[0]

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts via the OpenAI API.

        Splits the input into chunks of max_batch_size and concatenates
        results. Preserves input order even if the API returns shuffled
        indices.

        Args:
            texts: List of text strings to embed.

        Returns:
            EmbeddingResult with embeddings in the same order as input.
        """
        if not texts:
            return EmbeddingResult(embeddings=[], total_tokens=0)

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        for start in range(0, len(texts), self._max_batch_size):
            batch = texts[start : start + self._max_batch_size]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            # Sort by index to preserve order within the batch
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
            total_tokens += response.usage.total_tokens

        return EmbeddingResult(
            embeddings=all_embeddings,
            total_tokens=total_tokens,
        )
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_embeddings.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/vector/embeddings.py tests/test_vector/test_embeddings.py
git commit -m "feat(vector): implement OpenAIEmbeddings provider with batching

TDD Task 2/8 of 04-vector-store plan.
Wraps AsyncOpenAI client with batch chunking, index-based ordering,
and configurable model/dimensions. All tests use mocked client."
```

---

## Task 3: Local Embeddings Provider (sentence-transformers fallback)

**Files:**
- `tests/test_vector/test_embeddings.py` (extend)
- `nemesis/vector/embeddings.py` (extend)

### Step 1 — Write failing test

```python
# tests/test_vector/test_embeddings.py — APPEND to existing file
"""Tests for local sentence-transformers embedding provider."""
import numpy as np
from unittest.mock import MagicMock, patch

from nemesis.vector.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    LocalEmbeddings,
)


class TestLocalEmbeddings:
    """Tests for the local sentence-transformers embedding provider."""

    @pytest.fixture
    def mock_st_model(self) -> MagicMock:
        """Create a mock SentenceTransformer model."""
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 384
        return model

    @pytest.fixture
    def provider(self, mock_st_model: MagicMock) -> LocalEmbeddings:
        """Create a LocalEmbeddings with a mocked model."""
        return LocalEmbeddings(model=mock_st_model)

    def test_default_properties(self, provider: LocalEmbeddings) -> None:
        assert provider.dimensions == 384
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_is_embedding_provider(self, provider: LocalEmbeddings) -> None:
        assert isinstance(provider, EmbeddingProvider)

    def test_custom_model_name(self, mock_st_model: MagicMock) -> None:
        provider = LocalEmbeddings(
            model=mock_st_model, model_name="custom-model"
        )
        assert provider.model_name == "custom-model"

    @pytest.mark.asyncio
    async def test_embed_single(
        self, provider: LocalEmbeddings, mock_st_model: MagicMock
    ) -> None:
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        result = await provider.embed_single("hello")
        assert result == pytest.approx([0.1, 0.2, 0.3])
        mock_st_model.encode.assert_called_once_with(
            ["hello"], normalize_embeddings=True, show_progress_bar=False
        )

    @pytest.mark.asyncio
    async def test_embed_batch(
        self, provider: LocalEmbeddings, mock_st_model: MagicMock
    ) -> None:
        mock_st_model.encode.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        result = await provider.embed(["a", "b", "c"])
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        assert result.embeddings[0] == pytest.approx([0.1, 0.2])
        assert result.embeddings[2] == pytest.approx([0.5, 0.6])
        # Local models don't report token usage
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_embed_empty_list(
        self, provider: LocalEmbeddings, mock_st_model: MagicMock
    ) -> None:
        result = await provider.embed([])
        assert result.embeddings == []
        mock_st_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_returns_plain_lists(
        self, provider: LocalEmbeddings, mock_st_model: MagicMock
    ) -> None:
        """Ensure numpy arrays are converted to plain Python lists."""
        mock_st_model.encode.return_value = np.array(
            [[0.1, 0.2]], dtype=np.float32
        )
        result = await provider.embed(["test"])
        assert isinstance(result.embeddings[0], list)
        assert isinstance(result.embeddings[0][0], float)
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_embeddings.py::TestLocalEmbeddings -x -v 2>&1 | head -20
```

### Step 3 — Implement

Append to `nemesis/vector/embeddings.py`:

```python
# Add at top with other imports:
import asyncio

# Append to end of file:

class LocalEmbeddings:
    """Local embedding provider using sentence-transformers.

    Fallback when no OpenAI API key is available. Uses
    all-MiniLM-L6-v2 (384 dimensions) by default.

    Args:
        model: A SentenceTransformer model instance.
        model_name: Display name for the model.
    """

    def __init__(
        self,
        model: object | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._model = model
        self._model_name = model_name
        if model is not None:
            self._dimensions = model.get_sentence_embedding_dimension()
        else:
            self._dimensions = 384

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Name of the sentence-transformers model."""
        return self._model_name

    def _ensure_model(self) -> None:
        """Lazy-load the sentence-transformers model if not provided."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install nemesis-ai[local-embeddings]"
                ) from e
            self._model = SentenceTransformer(self._model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A single embedding vector.
        """
        result = await self.embed([text])
        return result.embeddings[0]

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts using sentence-transformers.

        Runs the model in a thread executor to avoid blocking the
        async event loop.

        Args:
            texts: List of text strings to embed.

        Returns:
            EmbeddingResult with embeddings. total_tokens is always 0
            since local models don't track token usage.
        """
        if not texts:
            return EmbeddingResult(embeddings=[], total_tokens=0)

        self._ensure_model()

        loop = asyncio.get_event_loop()
        embeddings_np = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ),
        )

        # Convert numpy arrays to plain Python lists of floats
        embeddings = [
            [float(x) for x in row] for row in embeddings_np
        ]

        return EmbeddingResult(embeddings=embeddings, total_tokens=0)
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_vector/test_embeddings.py -x -v
```

### Step 5 — Commit

```bash
git add nemesis/vector/embeddings.py tests/test_vector/test_embeddings.py
git commit -m "feat(vector): implement LocalEmbeddings with sentence-transformers

TDD Task 3/8 of 04-vector-store plan.
Fallback provider for environments without OpenAI API key.
Uses all-MiniLM-L6-v2 (384 dims), lazy-loads model, runs
encoding in thread executor to keep async loop free."
```

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

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | EmbeddingProvider protocol + EmbeddingResult | `embeddings.py` | 5 |
| 2 | OpenAIEmbeddings with batching + ordering | `embeddings.py` | 7 |
| 3 | LocalEmbeddings with sentence-transformers | `embeddings.py` | 6 |
| 4 | SearchResult model + VectorStore init | `store.py` | 8 |
| 5 | VectorStore.add() with validation | `store.py` | 6 |
| 6 | VectorStore.search() with metadata filter | `store.py` | 8 |
| 7 | VectorStore.delete() + delete_by_file() | `store.py` | 7 |
| 8 | Factory function + integration tests | `__init__.py` | 6 |
| **Total** | | **4 files** | **53 tests** |

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
