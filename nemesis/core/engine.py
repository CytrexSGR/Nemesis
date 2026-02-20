"""NemesisEngine -- central orchestrator that wires all components."""

from __future__ import annotations

import asyncio
from typing import Any

from nemesis.core.config import NemesisConfig
from nemesis.core.hooks import HookManager
from nemesis.graph import create_graph_adapter
from nemesis.indexer.pipeline import IndexingPipeline
from nemesis.memory.context import SessionContext
from nemesis.memory.conventions import ConventionManager
from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.rules import RulesManager
from nemesis.parser.bridge import ParserBridge
from nemesis.vector import create_embedding_provider
from nemesis.vector.store import VectorStore


class SyncEmbeddingWrapper:
    """Wraps an async EmbeddingProvider for synchronous usage."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def embed(self, texts: list[str]) -> Any:
        return self._get_loop().run_until_complete(self._provider.embed(texts))

    def embed_single(self, text: str) -> list[float]:
        return self._get_loop().run_until_complete(self._provider.embed_single(text))

    def close(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.close()


class SyncVectorStoreWrapper:
    """Wraps an async VectorStore for synchronous usage."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def initialize(self, dimensions: int) -> None:
        return self._get_loop().run_until_complete(self._store.initialize(dimensions))

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        return self._get_loop().run_until_complete(
            self._store.add(ids, texts, embeddings, metadata)
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        return self._get_loop().run_until_complete(
            self._store.search(query_vector, limit, filter_metadata)
        )

    def delete(self, ids: list[str]) -> None:
        return self._get_loop().run_until_complete(self._store.delete(ids))

    def delete_by_file(self, file_path: str) -> None:
        return self._get_loop().run_until_complete(self._store.delete_by_file(file_path))

    def delete_embeddings(self, ids: list[str]) -> None:
        """Alias for delete() -- used by delete_file_data in delta.py."""
        return self.delete(ids)

    def count(self) -> int:
        return self._get_loop().run_until_complete(self._store.count())

    def close(self) -> None:
        self._get_loop().run_until_complete(self._store.close())
        if self._loop and not self._loop.is_closed():
            self._loop.close()


class NemesisEngine:
    """Central engine that wires together all Nemesis components.

    Creates and manages graph adapter, vector store, embedding provider,
    parser, indexing pipeline, and memory managers from a NemesisConfig.
    """

    def __init__(self, config: NemesisConfig | None = None) -> None:
        self.config = config or NemesisConfig()
        self._graph = None
        self._vector_store = None
        self._embedder = None
        self._parser = None
        self._pipeline = None
        self._session = None
        self._rules = None
        self._decisions = None
        self._conventions = None
        self._hooks = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all components. Must be called before use."""
        if self._initialized:
            return

        cfg = self.config

        # Graph
        graph_kwargs: dict[str, Any] = {"db_path": str(cfg.project_root / cfg.graph_path)}
        if cfg.graph_backend == "neo4j":
            graph_kwargs = {
                "uri": cfg.neo4j_uri,
                "user": cfg.neo4j_user,
                "password": cfg.neo4j_password,
            }
        self._graph = create_graph_adapter(
            backend=cfg.graph_backend,
            create_schema=True,
            **graph_kwargs,
        )

        # Vector Store (async -> sync wrapped)
        raw_store = VectorStore(path=str(cfg.project_root / cfg.vector_path))
        self._vector_store = SyncVectorStoreWrapper(raw_store)

        # Embedder (async -> sync wrapped)
        embed_kwargs: dict[str, Any] = {}
        if cfg.vector_provider == "openai":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=cfg.openai_api_key or None)
            embed_kwargs["client"] = client
            embed_kwargs["model"] = cfg.vector_model
        raw_embedder = create_embedding_provider(
            provider_type=cfg.vector_provider, **embed_kwargs
        )
        self._embedder = SyncEmbeddingWrapper(raw_embedder)

        # Initialize vector store with embedding dimensions
        self._vector_store.initialize(raw_embedder.dimensions)

        # Parser
        self._parser = ParserBridge()

        # Pipeline
        self._pipeline = IndexingPipeline(
            parser=self._parser,
            graph=self._graph,
            vector_store=self._vector_store,
            embedder=self._embedder,
        )

        # Memory
        self._session = SessionContext()
        self._rules = RulesManager(self._graph)
        self._decisions = DecisionsManager(self._graph)
        self._conventions = ConventionManager(self._graph)

        # Hooks
        self._hooks = HookManager()

        self._initialized = True

    @property
    def graph(self):
        self._ensure_initialized()
        return self._graph

    @property
    def vector_store(self):
        self._ensure_initialized()
        return self._vector_store

    @property
    def embedder(self):
        self._ensure_initialized()
        return self._embedder

    @property
    def parser(self):
        self._ensure_initialized()
        return self._parser

    @property
    def pipeline(self):
        self._ensure_initialized()
        return self._pipeline

    @property
    def session(self):
        self._ensure_initialized()
        return self._session

    @property
    def rules(self):
        self._ensure_initialized()
        return self._rules

    @property
    def decisions(self):
        self._ensure_initialized()
        return self._decisions

    @property
    def conventions(self):
        self._ensure_initialized()
        return self._conventions

    @property
    def hooks(self):
        self._ensure_initialized()
        return self._hooks

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("NemesisEngine not initialized. Call initialize() first.")

    def close(self) -> None:
        """Close all resources."""
        if self._hooks:
            self._hooks.clear()
        if self._graph:
            self._graph.close()
        if self._vector_store:
            self._vector_store.close()
        if self._embedder:
            self._embedder.close()
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *exc):
        self.close()
