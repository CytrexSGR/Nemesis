"""Tests for embedding provider protocol and data models."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
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
    def provider(self, mock_openai_client: MagicMock):
        """Create an OpenAIEmbeddings with a mocked client."""
        from nemesis.vector.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(client=mock_openai_client)

    def test_default_properties(self, provider) -> None:
        assert provider.dimensions == 1536
        assert provider.model_name == "text-embedding-3-small"

    def test_custom_model(self, mock_openai_client: MagicMock) -> None:
        from nemesis.vector.embeddings import OpenAIEmbeddings

        provider = OpenAIEmbeddings(
            client=mock_openai_client,
            model="text-embedding-3-large",
            dimensions=3072,
        )
        assert provider.dimensions == 3072
        assert provider.model_name == "text-embedding-3-large"

    def test_is_embedding_provider(self, provider) -> None:
        assert isinstance(provider, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_embed_single(self, provider, mock_openai_client: MagicMock) -> None:
        mock_openai_client.embeddings.create.return_value = self._make_mock_response(
            [[0.1, 0.2, 0.3]], total_tokens=5
        )
        result = await provider.embed_single("hello world")
        assert result == [0.1, 0.2, 0.3]
        mock_openai_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["hello world"],
            dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider, mock_openai_client: MagicMock) -> None:
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_openai_client.embeddings.create.return_value = self._make_mock_response(
            embeddings, total_tokens=15
        )
        result = await provider.embed(["text1", "text2", "text3"])
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        assert result.total_tokens == 15
        assert result.embeddings[0] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, provider, mock_openai_client: MagicMock) -> None:
        result = await provider.embed([])
        assert result.embeddings == []
        assert result.total_tokens == 0
        mock_openai_client.embeddings.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embed_batch_chunking(self, mock_openai_client: MagicMock) -> None:
        """Large batches should be split into chunks of max_batch_size."""
        from nemesis.vector.embeddings import OpenAIEmbeddings

        provider = OpenAIEmbeddings(client=mock_openai_client, max_batch_size=2)
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
    async def test_embed_preserves_order(self, provider, mock_openai_client: MagicMock) -> None:
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


class TestLocalEmbeddings:
    """Tests for the local sentence-transformers embedding provider."""

    @pytest.fixture
    def mock_st_model(self) -> MagicMock:
        """Create a mock SentenceTransformer model."""
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 384
        return model

    @pytest.fixture
    def provider(self, mock_st_model: MagicMock):
        """Create a LocalEmbeddings with a mocked model."""
        from nemesis.vector.embeddings import LocalEmbeddings

        return LocalEmbeddings(model=mock_st_model)

    def test_default_properties(self, provider) -> None:
        assert provider.dimensions == 384
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_is_embedding_provider(self, provider) -> None:
        assert isinstance(provider, EmbeddingProvider)

    def test_custom_model_name(self, mock_st_model: MagicMock) -> None:
        from nemesis.vector.embeddings import LocalEmbeddings

        provider = LocalEmbeddings(model=mock_st_model, model_name="custom-model")
        assert provider.model_name == "custom-model"

    @pytest.mark.asyncio
    async def test_embed_single(self, provider, mock_st_model: MagicMock) -> None:
        mock_st_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        result = await provider.embed_single("hello")
        assert result == pytest.approx([0.1, 0.2, 0.3])
        mock_st_model.encode.assert_called_once_with(
            ["hello"], normalize_embeddings=True, show_progress_bar=False
        )

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider, mock_st_model: MagicMock) -> None:
        mock_st_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = await provider.embed(["a", "b", "c"])
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        assert result.embeddings[0] == pytest.approx([0.1, 0.2])
        assert result.embeddings[2] == pytest.approx([0.5, 0.6])
        # Local models don't report token usage
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, provider, mock_st_model: MagicMock) -> None:
        result = await provider.embed([])
        assert result.embeddings == []
        mock_st_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_returns_plain_lists(self, provider, mock_st_model: MagicMock) -> None:
        """Ensure numpy arrays are converted to plain Python lists."""
        mock_st_model.encode.return_value = np.array([[0.1, 0.2]], dtype=np.float32)
        result = await provider.embed(["test"])
        assert isinstance(result.embeddings[0], list)
        assert isinstance(result.embeddings[0][0], float)
