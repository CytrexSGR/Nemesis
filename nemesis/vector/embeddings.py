"""Embedding providers for Nemesis vector store.

Supports OpenAI text-embedding-3-small (default) and local
sentence-transformers as fallback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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

        Convenience method -- equivalent to embed([text]).embeddings[0].

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
            self._dimensions: int = model.get_sentence_embedding_dimension()
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
            lambda: self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        )

        # Convert numpy arrays to plain Python lists of floats
        embeddings = [[float(x) for x in row] for row in embeddings_np]

        return EmbeddingResult(embeddings=embeddings, total_tokens=0)
