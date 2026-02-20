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
        raise ValueError(f"Unknown provider type: {provider_type!r}. Supported: 'openai', 'local'")


__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "LocalEmbeddings",
    "OpenAIEmbeddings",
    "SearchResult",
    "VectorStore",
    "create_embedding_provider",
]
