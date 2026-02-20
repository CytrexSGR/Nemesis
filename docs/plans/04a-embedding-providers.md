# 04a — Embedding Providers

> **Arbeitspaket D1** — Teil 1 von 3 des Vector Store Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Embedding-Protokoll und zwei Provider (OpenAI + lokaler Fallback) implementieren. Umfasst Tasks 1, 2, 3 des ursprünglichen Vector-Store-Plans.

**Architecture:** `EmbeddingProvider` Protocol mit zwei Implementierungen (OpenAI, local sentence-transformers). Tests nutzen gemockte OpenAI-Clients.

**Tech Stack:** OpenAI API (text-embedding-3-small), sentence-transformers (all-MiniLM-L6-v2), numpy

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [01-project-setup](01-project-setup.md)

**Navigation:** [Nächstes Paket: D2 — Vector Store Core →](04b-vector-store-core.md)

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

## Summary

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 1 | EmbeddingProvider Protocol + EmbeddingResult | `embeddings.py` | 5 |
| 2 | OpenAIEmbeddings mit Batching + Ordering | `embeddings.py` | 7 |
| 3 | LocalEmbeddings mit sentence-transformers | `embeddings.py` | 6 |
| **Gesamt** | | **2 Dateien** | **18 Tests** |

### Dependencies (pyproject.toml additions)

```toml
[project]
dependencies = [
    # ... existing ...
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

**Navigation:** [Nächstes Paket: D2 — Vector Store Core →](04b-vector-store-core.md)
