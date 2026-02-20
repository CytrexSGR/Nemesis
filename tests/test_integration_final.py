"""End-to-end integration tests — verify wiring between Nemesis subsystems."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from nemesis.core.config import NemesisConfig
from nemesis.core.hooks import HookEvent, HookManager
from nemesis.core.server import TOOL_DEFINITIONS, create_mcp_server
from nemesis.tools.tools import get_memory, remember_rule, search_code

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_engine(tmp_path):
    """Create a NemesisEngine where external backends are fully mocked.

    After initialize(), the sync wrappers (vector_store, embedder) are
    replaced with plain MagicMocks so tests never hit asyncio event loops.
    """
    config = NemesisConfig(
        data_dir=tmp_path,
        openai_api_key="sk-fake",
    )

    with (
        patch("nemesis.core.engine.create_graph_adapter") as mock_graph_factory,
        patch("nemesis.core.engine.VectorStore") as mock_vs_cls,
        patch("nemesis.core.engine.create_embedding_provider") as mock_embed_cls,
        patch("nemesis.core.engine.ParserBridge"),
    ):
        mock_graph = MagicMock()
        mock_graph.query.return_value = []
        mock_graph_factory.return_value = mock_graph
        mock_vs_cls.return_value = AsyncMock()
        mock_embed_cls.return_value = AsyncMock()

        from nemesis.core.engine import NemesisEngine

        engine = NemesisEngine(config)
        engine.initialize()

    # Replace sync wrappers with plain mocks to avoid asyncio issues
    mock_vector = MagicMock()
    mock_vector.search.return_value = []
    mock_vector.close = MagicMock()
    engine._vector_store = mock_vector

    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.1, 0.2, 0.3]
    mock_embedder.close = MagicMock()
    engine._embedder = mock_embedder

    return engine


# ===========================================================================
# TestEngineHooksIntegration
# ===========================================================================


class TestEngineHooksIntegration:
    """Verify HookManager is properly wired into NemesisEngine."""

    def test_engine_has_hooks_after_init(self, tmp_path):
        """engine.hooks is a HookManager instance after initialize()."""
        engine = _make_mock_engine(tmp_path)
        try:
            assert isinstance(engine.hooks, HookManager)
        finally:
            engine.close()

    def test_hooks_cleared_on_close(self, tmp_path):
        """After close(), all hooks are removed (hook_count == 0)."""
        engine = _make_mock_engine(tmp_path)

        # Register a dummy hook so count > 0
        engine.hooks.register(HookEvent.PRE_INDEX, lambda ctx: None)
        assert engine.hooks.hook_count > 0

        engine.close()
        # hooks object still exists but all hooks cleared
        assert engine._hooks.hook_count == 0

    def test_hook_receives_events(self, tmp_path):
        """A registered PRE_INDEX hook receives the emitted event."""
        engine = _make_mock_engine(tmp_path)
        received = []

        def on_pre_index(ctx):
            received.append(ctx)

        try:
            engine.hooks.register(HookEvent.PRE_INDEX, on_pre_index)
            engine.hooks.emit(HookEvent.PRE_INDEX, {"file": "test.py"})

            assert len(received) == 1
            assert received[0].event == HookEvent.PRE_INDEX
            assert received[0].data["file"] == "test.py"
        finally:
            engine.close()


# ===========================================================================
# TestToolsWithEngine
# ===========================================================================


class TestToolsWithEngine:
    """Verify tool functions work with a (mocked) engine."""

    def test_search_code_end_to_end(self, tmp_path):
        """search_code returns expected structure through engine wiring."""
        engine = _make_mock_engine(tmp_path)

        # Mock the vector_store.search to return a result
        mock_result = MagicMock()
        mock_result.id = "chunk-1"
        mock_result.text = "def hello(): pass"
        mock_result.score = 0.95
        mock_result.metadata = {
            "file": "hello.py",
            "start_line": 1,
            "end_line": 1,
            "parent_node_id": None,
        }
        engine.vector_store.search.return_value = [mock_result]

        try:
            result = search_code(engine, query="hello function", limit=5)

            assert result["query"] == "hello function"
            assert result["count"] == 1
            assert len(result["results"]) == 1
            assert result["results"][0]["file"] == "hello.py"
            assert result["results"][0]["score"] == 0.95
            # Session should have recorded the query
            assert "hello function" in engine.session.get_queries()
        finally:
            engine.close()

    def test_remember_rule_persists(self, tmp_path):
        """remember_rule creates a Rule that get_memory can retrieve."""
        engine = _make_mock_engine(tmp_path)

        try:
            # remember_rule calls engine.rules.add_rule which calls graph.add_node
            rule_result = remember_rule(engine, content="Use type hints everywhere")

            assert rule_result["content"] == "Use type hints everywhere"
            assert rule_result["scope"] == "project"
            assert "id" in rule_result

            # Verify graph.add_node was called
            engine.graph.add_node.assert_called_once()
            node_arg = engine.graph.add_node.call_args[0][0]
            assert node_arg.node_type == "Rule"
            assert node_arg.properties["content"] == "Use type hints everywhere"

            # get_memory queries the graph — configure return values
            # Rules query returns our rule, decisions and conventions return empty
            def query_side_effect(cypher, **kwargs):
                if "Rule" in cypher:
                    return [
                        {
                            "r": {
                                "id": rule_result["id"],
                                "content": "Use type hints everywhere",
                                "scope": "project",
                                "source": "user",
                                "created_at": rule_result["created_at"],
                            }
                        }
                    ]
                return []

            engine.graph.query.side_effect = query_side_effect

            memory = get_memory(engine)
            assert memory["total"] >= 1
            assert any(
                r["content"] == "Use type hints everywhere" for r in memory["rules"]
            )
        finally:
            engine.close()


# ===========================================================================
# TestCLIWithMocks
# ===========================================================================


class TestCLIWithMocks:
    """Verify CLI commands dispatch through engine correctly."""

    def test_index_and_query_flow(self, tmp_path):
        """Mock engine, run index + query CLI commands in sequence."""
        from click.testing import CliRunner

        from nemesis.core.cli import main
        from nemesis.indexer.models import IndexResult

        runner = CliRunner()

        mock_result = IndexResult(
            files_indexed=3,
            nodes_created=10,
            edges_created=5,
            chunks_created=8,
            embeddings_created=8,
            duration_ms=42.0,
            errors=[],
        )

        # Patch NemesisEngine so CLI doesn't need real backends
        with patch("nemesis.core.cli.NemesisEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine
            mock_engine.__enter__ = MagicMock(return_value=mock_engine)
            mock_engine.__exit__ = MagicMock(return_value=False)
            mock_engine.pipeline.index_project.return_value = mock_result
            mock_engine.config = NemesisConfig(data_dir=tmp_path)

            # Index command
            result = runner.invoke(main, ["index", str(tmp_path)])
            assert result.exit_code == 0, f"index failed: {result.output}"
            assert "Indexed 3 files" in result.output
            assert "Nodes: 10" in result.output

            # Query command — mock embedder + vector_store
            mock_search_result = MagicMock()
            mock_search_result.score = 0.88
            mock_search_result.text = "class Foo: pass"
            mock_search_result.metadata = {
                "file": "foo.py",
                "start_line": 1,
                "end_line": 1,
            }
            mock_engine.embedder.embed_single.return_value = [0.1, 0.2]
            mock_engine.vector_store.search.return_value = [mock_search_result]

            result = runner.invoke(
                main, ["query", "find Foo", "-p", str(tmp_path)]
            )
            assert result.exit_code == 0, f"query failed: {result.output}"
            assert "0.88" in result.output
            assert "foo.py" in result.output


# ===========================================================================
# TestServerToolRegistration
# ===========================================================================


class TestServerToolRegistration:
    """Verify MCP server correctly registers and dispatches tools."""

    def test_all_tools_registered_in_server(self, tmp_path):
        """create_mcp_server registers all 8 tools."""
        engine = _make_mock_engine(tmp_path)

        try:
            server = create_mcp_server(engine)

            # The server has TOOL_DEFINITIONS with 10 entries
            expected_names = {td["name"] for td in TOOL_DEFINITIONS}
            assert len(expected_names) == 10
            assert expected_names == {
                "search_code",
                "get_context",
                "index_project",
                "update_project",
                "remember_rule",
                "remember_decision",
                "get_memory",
                "get_session_summary",
                "list_projects",
                "remove_project",
            }

            # Verify the server object was created
            assert server is not None
            assert server.name == "nemesis"
        finally:
            engine.close()

    def test_server_tool_calls_tool_function(self, tmp_path):
        """call_tool for search_code dispatches to the actual tool function."""
        engine = _make_mock_engine(tmp_path)

        # Prepare mock for vector search
        mock_result = MagicMock()
        mock_result.id = "c1"
        mock_result.text = "x = 1"
        mock_result.score = 0.9
        mock_result.metadata = {"file": "a.py", "start_line": 1, "end_line": 1}
        engine.vector_store.search.return_value = [mock_result]

        try:
            # Verify the dispatch table maps to the correct function
            from nemesis.core.server import _TOOL_DISPATCH

            func = _TOOL_DISPATCH["search_code"]
            result = func(engine, query="test query", limit=5)

            assert result["query"] == "test query"
            assert result["count"] == 1
            assert result["results"][0]["file"] == "a.py"

            # Verify embedder was called
            engine.embedder.embed_single.assert_called_with("test query")
            # Verify vector_store.search was called
            engine.vector_store.search.assert_called_once()
        finally:
            engine.close()
