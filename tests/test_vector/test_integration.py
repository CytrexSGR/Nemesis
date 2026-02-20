"""Integration tests for the vector module.

Tests the full flow: embed texts -> store in LanceDB -> search.
Uses mocked OpenAI client + real (temporary) LanceDB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemesis.vector import create_embedding_provider
from nemesis.vector.embeddings import OpenAIEmbeddings
from nemesis.vector.store import VectorStore

if TYPE_CHECKING:
    from pathlib import Path


class TestCreateEmbeddingProvider:
    """Tests for the factory function."""

    def test_create_openai_provider(self) -> None:
        mock_client = MagicMock()
        provider = create_embedding_provider(provider_type="openai", client=mock_client)
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
        provider = create_embedding_provider(provider_type="local", model=mock_model)
        assert provider.dimensions == 384

    def test_create_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_embedding_provider(provider_type="unknown")


class TestEmbedAndStore:
    """End-to-end: embed -> store -> search with mocked embeddings."""

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock()
        return client

    @pytest.fixture
    def mock_provider(self, mock_openai_client: MagicMock) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(client=mock_openai_client, dimensions=4)

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
        mock_openai_client.embeddings.create.return_value = self._make_mock_response(
            chunk_embeddings, total_tokens=30
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
        mock_openai_client.embeddings.create.return_value = self._make_mock_response(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            total_tokens=20,
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

        mock_openai_client.embeddings.create.return_value = self._make_mock_response(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 0.5, 0.5],
            ],
            total_tokens=30,
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
        results = await store.search(query_embedding=[0.5, 0.5, 0.0, 0.0], limit=1)
        assert results[0].metadata["version"] == 2

        await store.close()
