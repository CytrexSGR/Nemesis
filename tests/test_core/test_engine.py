"""Tests for NemesisEngine and sync wrappers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemesis.core.config import NemesisConfig
from nemesis.core.engine import (
    NemesisEngine,
    SyncEmbeddingWrapper,
    SyncVectorStoreWrapper,
)

# ---------------------------------------------------------------
# TestSyncEmbeddingWrapper
# ---------------------------------------------------------------


class TestSyncEmbeddingWrapper:
    """Tests for the SyncEmbeddingWrapper."""

    def test_embed_calls_async_provider(self):
        """embed() delegates to the async provider's embed method."""
        mock_provider = AsyncMock()
        mock_provider.embed.return_value = MagicMock(embeddings=[[0.1, 0.2]])
        wrapper = SyncEmbeddingWrapper(mock_provider)

        result = wrapper.embed(["hello"])

        mock_provider.embed.assert_awaited_once_with(["hello"])
        assert result.embeddings == [[0.1, 0.2]]
        wrapper.close()

    def test_embed_single_calls_async_provider(self):
        """embed_single() delegates to the async provider's embed_single method."""
        mock_provider = AsyncMock()
        mock_provider.embed_single.return_value = [0.3, 0.4, 0.5]
        wrapper = SyncEmbeddingWrapper(mock_provider)

        result = wrapper.embed_single("world")

        mock_provider.embed_single.assert_awaited_once_with("world")
        assert result == [0.3, 0.4, 0.5]
        wrapper.close()

    def test_close_closes_loop(self):
        """close() shuts down the internal event loop."""
        mock_provider = AsyncMock()
        mock_provider.embed.return_value = MagicMock(embeddings=[])
        wrapper = SyncEmbeddingWrapper(mock_provider)

        # Force loop creation
        wrapper.embed([])
        loop = wrapper._loop
        assert loop is not None
        assert not loop.is_closed()

        wrapper.close()
        assert loop.is_closed()


# ---------------------------------------------------------------
# TestSyncVectorStoreWrapper
# ---------------------------------------------------------------


class TestSyncVectorStoreWrapper:
    """Tests for the SyncVectorStoreWrapper."""

    def test_add_calls_async_store(self):
        """add() delegates to the async store's add method."""
        mock_store = AsyncMock()
        wrapper = SyncVectorStoreWrapper(mock_store)

        ids = ["id1"]
        texts = ["text1"]
        embeddings = [[0.1, 0.2]]
        metadata = [{"file": "test.py"}]
        wrapper.add(ids, texts, embeddings, metadata)

        mock_store.add.assert_awaited_once_with(ids, texts, embeddings, metadata, project_id="")
        wrapper.close()

    def test_search_calls_async_store(self):
        """search() delegates to the async store's search method."""
        mock_store = AsyncMock()
        mock_store.search.return_value = []
        wrapper = SyncVectorStoreWrapper(mock_store)

        result = wrapper.search([0.1, 0.2], limit=5)

        mock_store.search.assert_awaited_once_with([0.1, 0.2], 5, None, project_id=None)
        assert result == []
        wrapper.close()

    def test_delete_calls_async_store(self):
        """delete() delegates to the async store's delete method."""
        mock_store = AsyncMock()
        wrapper = SyncVectorStoreWrapper(mock_store)

        wrapper.delete(["id1", "id2"])

        mock_store.delete.assert_awaited_once_with(["id1", "id2"])
        wrapper.close()

    def test_delete_embeddings_alias(self):
        """delete_embeddings() is an alias for delete()."""
        mock_store = AsyncMock()
        wrapper = SyncVectorStoreWrapper(mock_store)

        wrapper.delete_embeddings(["id3"])

        mock_store.delete.assert_awaited_once_with(["id3"])
        wrapper.close()

    def test_initialize_calls_async_store(self):
        """initialize() delegates to the async store's initialize method."""
        mock_store = AsyncMock()
        wrapper = SyncVectorStoreWrapper(mock_store)

        wrapper.initialize(1536)

        mock_store.initialize.assert_awaited_once_with(1536)
        wrapper.close()

    def test_count_calls_async_store(self):
        """count() delegates to the async store's count method."""
        mock_store = AsyncMock()
        mock_store.count.return_value = 42
        wrapper = SyncVectorStoreWrapper(mock_store)

        result = wrapper.count()

        mock_store.count.assert_awaited_once()
        assert result == 42
        wrapper.close()

    def test_close_closes_store_and_loop(self):
        """close() calls async store close and shuts down the loop."""
        mock_store = AsyncMock()
        wrapper = SyncVectorStoreWrapper(mock_store)

        # Force loop creation
        wrapper.count()
        loop = wrapper._loop
        assert loop is not None

        wrapper.close()

        mock_store.close.assert_awaited_once()
        assert loop.is_closed()


# ---------------------------------------------------------------
# TestNemesisEngine
# ---------------------------------------------------------------


class TestNemesisEngine:
    """Tests for the NemesisEngine."""

    def test_default_config(self):
        """Engine creates with default NemesisConfig when none given."""
        engine = NemesisEngine()
        assert isinstance(engine.config, NemesisConfig)
        assert engine.config.project_name == "nemesis"

    def test_custom_config(self):
        """Engine uses the provided config."""
        cfg = NemesisConfig(project_name="custom")
        engine = NemesisEngine(config=cfg)
        assert engine.config.project_name == "custom"

    def test_not_initialized_raises(self):
        """Accessing properties before initialize() raises RuntimeError."""
        engine = NemesisEngine()
        properties = [
            "graph",
            "vector_store",
            "embedder",
            "parser",
            "pipeline",
            "session",
            "rules",
            "decisions",
            "conventions",
        ]
        for prop_name in properties:
            with pytest.raises(RuntimeError, match="not initialized"):
                getattr(engine, prop_name)

    @patch("nemesis.core.engine.ConventionManager")
    @patch("nemesis.core.engine.DecisionsManager")
    @patch("nemesis.core.engine.RulesManager")
    @patch("nemesis.core.engine.SessionContext")
    @patch("nemesis.core.engine.IndexingPipeline")
    @patch("nemesis.core.engine.ParserBridge")
    @patch("nemesis.core.engine.create_embedding_provider")
    @patch("nemesis.core.engine.VectorStore")
    @patch("nemesis.core.engine.create_graph_adapter")
    @patch("openai.AsyncOpenAI")
    def test_initialize_creates_components(
        self,
        _mock_openai_client,
        mock_graph_adapter,
        mock_vector_store,
        mock_embed_provider,
        mock_parser,
        mock_pipeline,
        mock_session,
        mock_rules,
        mock_decisions,
        mock_conventions,
    ):
        """After initialize(), all properties are available."""
        mock_graph_adapter.return_value = MagicMock()
        mock_vector_store.return_value = AsyncMock()
        mock_embed_provider.return_value = AsyncMock()
        mock_parser.return_value = MagicMock()

        engine = NemesisEngine()
        engine.initialize()

        assert engine._initialized is True
        assert engine.graph is not None
        assert engine.vector_store is not None
        assert engine.embedder is not None
        assert engine.parser is not None
        assert engine.pipeline is not None
        assert engine.session is not None
        assert engine.rules is not None
        assert engine.decisions is not None
        assert engine.conventions is not None

        mock_graph_adapter.assert_called_once()
        mock_vector_store.assert_called_once()
        mock_embed_provider.assert_called_once()
        mock_parser.assert_called_once()
        mock_pipeline.assert_called_once()

    @patch("nemesis.core.engine.ConventionManager")
    @patch("nemesis.core.engine.DecisionsManager")
    @patch("nemesis.core.engine.RulesManager")
    @patch("nemesis.core.engine.SessionContext")
    @patch("nemesis.core.engine.IndexingPipeline")
    @patch("nemesis.core.engine.ParserBridge")
    @patch("nemesis.core.engine.create_embedding_provider")
    @patch("nemesis.core.engine.VectorStore")
    @patch("nemesis.core.engine.create_graph_adapter")
    @patch("openai.AsyncOpenAI")
    def test_context_manager(
        self,
        _mock_openai_client,
        mock_graph_adapter,
        mock_vector_store,
        mock_embed_provider,
        mock_parser,
        mock_pipeline,
        mock_session,
        mock_rules,
        mock_decisions,
        mock_conventions,
    ):
        """Context manager calls initialize on enter and close on exit."""
        mock_graph = MagicMock()
        mock_graph_adapter.return_value = mock_graph
        mock_vector_store.return_value = AsyncMock()
        mock_embed_provider.return_value = AsyncMock()

        with NemesisEngine() as engine:
            assert engine._initialized is True
            assert engine.graph is not None

        assert engine._initialized is False

    @patch("nemesis.core.engine.ConventionManager")
    @patch("nemesis.core.engine.DecisionsManager")
    @patch("nemesis.core.engine.RulesManager")
    @patch("nemesis.core.engine.SessionContext")
    @patch("nemesis.core.engine.IndexingPipeline")
    @patch("nemesis.core.engine.ParserBridge")
    @patch("nemesis.core.engine.create_embedding_provider")
    @patch("nemesis.core.engine.VectorStore")
    @patch("nemesis.core.engine.create_graph_adapter")
    @patch("openai.AsyncOpenAI")
    def test_close_resets_state(
        self,
        _mock_openai_client,
        mock_graph_adapter,
        mock_vector_store,
        mock_embed_provider,
        mock_parser,
        mock_pipeline,
        mock_session,
        mock_rules,
        mock_decisions,
        mock_conventions,
    ):
        """After close(), initialized is False and graph.close() is called."""
        mock_graph = MagicMock()
        mock_graph_adapter.return_value = mock_graph
        mock_vector_store.return_value = AsyncMock()
        mock_embed_provider.return_value = AsyncMock()

        engine = NemesisEngine()
        engine.initialize()
        assert engine._initialized is True

        engine.close()
        assert engine._initialized is False
        mock_graph.close.assert_called_once()

    @patch("nemesis.core.engine.ConventionManager")
    @patch("nemesis.core.engine.DecisionsManager")
    @patch("nemesis.core.engine.RulesManager")
    @patch("nemesis.core.engine.SessionContext")
    @patch("nemesis.core.engine.IndexingPipeline")
    @patch("nemesis.core.engine.ParserBridge")
    @patch("nemesis.core.engine.create_embedding_provider")
    @patch("nemesis.core.engine.VectorStore")
    @patch("nemesis.core.engine.create_graph_adapter")
    @patch("openai.AsyncOpenAI")
    def test_double_initialize_noop(
        self,
        _mock_openai_client,
        mock_graph_adapter,
        mock_vector_store,
        mock_embed_provider,
        mock_parser,
        mock_pipeline,
        mock_session,
        mock_rules,
        mock_decisions,
        mock_conventions,
    ):
        """Calling initialize() twice is idempotent -- no second setup."""
        mock_graph_adapter.return_value = MagicMock()
        mock_vector_store.return_value = AsyncMock()
        mock_embed_provider.return_value = AsyncMock()

        engine = NemesisEngine()
        engine.initialize()
        engine.initialize()

        # Factories should only be called once
        mock_graph_adapter.assert_called_once()
        mock_vector_store.assert_called_once()
        mock_embed_provider.assert_called_once()

    @patch("nemesis.core.engine.ConventionManager")
    @patch("nemesis.core.engine.DecisionsManager")
    @patch("nemesis.core.engine.RulesManager")
    @patch("nemesis.core.engine.SessionContext")
    @patch("nemesis.core.engine.IndexingPipeline")
    @patch("nemesis.core.engine.ParserBridge")
    @patch("nemesis.core.engine.create_embedding_provider")
    @patch("nemesis.core.engine.VectorStore")
    @patch("nemesis.core.engine.create_graph_adapter")
    @patch("openai.AsyncOpenAI")
    def test_neo4j_backend_kwargs(
        self,
        _mock_openai_client,
        mock_graph_adapter,
        mock_vector_store,
        mock_embed_provider,
        mock_parser,
        mock_pipeline,
        mock_session,
        mock_rules,
        mock_decisions,
        mock_conventions,
    ):
        """Neo4j backend passes URI/user/password to create_graph_adapter."""
        mock_graph_adapter.return_value = MagicMock()
        mock_vector_store.return_value = AsyncMock()
        mock_embed_provider.return_value = AsyncMock()

        cfg = NemesisConfig(
            graph_backend="neo4j",
            neo4j_uri="bolt://db:7687",
            neo4j_user="admin",
            neo4j_password="secret",
        )
        engine = NemesisEngine(config=cfg)
        engine.initialize()

        mock_graph_adapter.assert_called_once_with(
            backend="neo4j",
            create_schema=True,
            uri="bolt://db:7687",
            user="admin",
            password="secret",
        )
        engine.close()

    @patch("nemesis.core.engine.ConventionManager")
    @patch("nemesis.core.engine.DecisionsManager")
    @patch("nemesis.core.engine.RulesManager")
    @patch("nemesis.core.engine.SessionContext")
    @patch("nemesis.core.engine.IndexingPipeline")
    @patch("nemesis.core.engine.ParserBridge")
    @patch("nemesis.core.engine.create_embedding_provider")
    @patch("nemesis.core.engine.VectorStore")
    @patch("nemesis.core.engine.create_graph_adapter")
    @patch("openai.AsyncOpenAI")
    def test_openai_embed_kwargs(
        self,
        mock_openai_client_cls,
        mock_graph_adapter,
        mock_vector_store,
        mock_embed_provider,
        mock_parser,
        mock_pipeline,
        mock_session,
        mock_rules,
        mock_decisions,
        mock_conventions,
    ):
        """OpenAI embedding provider receives AsyncOpenAI client and model."""
        mock_graph_adapter.return_value = MagicMock()
        mock_vector_store.return_value = AsyncMock()
        mock_embed_provider.return_value = AsyncMock()
        mock_client_instance = MagicMock()
        mock_openai_client_cls.return_value = mock_client_instance

        cfg = NemesisConfig(
            vector_provider="openai",
            openai_api_key="sk-test",
            vector_model="text-embedding-3-large",
        )
        engine = NemesisEngine(config=cfg)
        engine.initialize()

        mock_openai_client_cls.assert_called_once_with(api_key="sk-test")
        mock_embed_provider.assert_called_once_with(
            provider_type="openai",
            client=mock_client_instance,
            model="text-embedding-3-large",
        )
        engine.close()
