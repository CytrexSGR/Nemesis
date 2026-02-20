"""Tests for nemesis.tools.tools — MCP tool implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from nemesis.graph.adapter import NodeData
from nemesis.indexer.models import IndexResult
from nemesis.memory.models import ConventionModel, DecisionModel, RuleModel
from nemesis.tools.tools import (
    get_context,
    get_memory,
    get_session_summary,
    index_project,
    remember_decision,
    remember_rule,
    search_code,
    update_project,
)
from nemesis.vector.store import SearchResult


def _make_engine(**overrides: object) -> MagicMock:
    """Create a mock NemesisEngine with common defaults."""
    engine = MagicMock()
    # Config defaults
    engine.config.languages = ["python"]
    engine.config.ignore_patterns = ["__pycache__", ".git"]
    return engine


# -------------------------------------------------------------------------
# TestSearchCode
# -------------------------------------------------------------------------


class TestSearchCode:
    def test_search_returns_results(self) -> None:
        """Mock embedder + vector_store, verify result structure."""
        engine = _make_engine()
        engine.embedder.embed_single.return_value = [0.1, 0.2, 0.3]
        engine.vector_store.search.return_value = [
            SearchResult(
                id="chunk-1",
                text="def foo(): pass",
                score=0.95,
                metadata={"file": "main.py", "start_line": 1, "end_line": 3},
            ),
            SearchResult(
                id="chunk-2",
                text="class Bar: ...",
                score=0.80,
                metadata={"file": "bar.py", "start_line": 10, "end_line": 20},
            ),
        ]

        result = search_code(engine, "find foo function", limit=5)

        assert result["query"] == "find foo function"
        assert result["count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["id"] == "chunk-1"
        assert result["results"][0]["file"] == "main.py"
        assert result["results"][0]["score"] == 0.95
        assert result["results"][1]["id"] == "chunk-2"

        engine.embedder.embed_single.assert_called_once_with("find foo function")
        engine.vector_store.search.assert_called_once_with(
            [0.1, 0.2, 0.3], limit=5
        )

    def test_search_enriches_from_graph(self) -> None:
        """When parent_node_id is present, graph.get_node is called."""
        engine = _make_engine()
        engine.embedder.embed_single.return_value = [0.1]
        engine.vector_store.search.return_value = [
            SearchResult(
                id="chunk-1",
                text="def foo(): pass",
                score=0.9,
                metadata={
                    "file": "main.py",
                    "parent_node_id": "node-func-foo",
                },
            ),
        ]
        engine.graph.get_node.return_value = NodeData(
            id="node-func-foo",
            node_type="Function",
            properties={"name": "foo"},
        )

        result = search_code(engine, "foo")

        engine.graph.get_node.assert_called_once_with("node-func-foo")
        item = result["results"][0]
        assert item["node_type"] == "Function"
        assert item["node_name"] == "foo"

    def test_search_records_session(self) -> None:
        """session.add_query and add_result are called."""
        engine = _make_engine()
        engine.embedder.embed_single.return_value = [0.0]
        engine.vector_store.search.return_value = []

        search_code(engine, "test query")

        engine.session.add_query.assert_called_once_with("test query")
        engine.session.add_result.assert_called_once()
        call_args = engine.session.add_result.call_args
        assert call_args[0][0] == "test query"
        assert call_args[0][1] == {"results": []}


# -------------------------------------------------------------------------
# TestGetContext
# -------------------------------------------------------------------------


class TestGetContext:
    def test_get_context_returns_nodes(self) -> None:
        """Mock graph.get_nodes_for_file, verify structure."""
        engine = _make_engine()
        engine.graph.get_nodes_for_file.return_value = [
            NodeData(
                id="node-1",
                node_type="Function",
                properties={"name": "foo", "start_line": 1, "end_line": 10},
            ),
        ]
        engine.graph.get_neighbors.return_value = []

        result = get_context(engine, "src/main.py")

        assert result["file"] == "src/main.py"
        assert result["node_count"] == 1
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "node-1"
        assert result["nodes"][0]["type"] == "Function"
        assert result["nodes"][0]["name"] == "foo"
        assert result["related"] == []

    def test_get_context_deduplicates_related(self) -> None:
        """Two nodes sharing the same neighbor should be deduplicated."""
        engine = _make_engine()
        shared_neighbor = NodeData(
            id="shared-dep",
            node_type="Class",
            properties={"name": "SharedDep"},
        )
        engine.graph.get_nodes_for_file.return_value = [
            NodeData(id="n1", node_type="Function", properties={"name": "a"}),
            NodeData(id="n2", node_type="Function", properties={"name": "b"}),
        ]
        # Both nodes return the same neighbor
        engine.graph.get_neighbors.return_value = [shared_neighbor]

        result = get_context(engine, "file.py")

        assert result["node_count"] == 2
        # shared-dep appears only once despite being returned for both nodes
        assert len(result["related"]) == 1
        assert result["related"][0]["id"] == "shared-dep"


# -------------------------------------------------------------------------
# TestIndexProject
# -------------------------------------------------------------------------


class TestIndexProject:
    def test_index_project_calls_pipeline(self) -> None:
        """Mock pipeline.index_project, verify kwargs."""
        engine = _make_engine()
        engine.pipeline.index_project.return_value = IndexResult(
            files_indexed=5,
            nodes_created=20,
            edges_created=15,
            chunks_created=30,
            embeddings_created=30,
            duration_ms=1234.5678,
            errors=[],
        )

        result = index_project(engine, "/my/project", languages=["python", "rust"])

        engine.pipeline.index_project.assert_called_once_with(
            Path("/my/project"),
            languages=["python", "rust"],
            ignore_dirs={"__pycache__", ".git"},
        )
        assert result["files_indexed"] == 5
        assert result["nodes_created"] == 20
        assert result["edges_created"] == 15
        assert result["chunks_created"] == 30
        assert result["embeddings_created"] == 30
        assert result["duration_ms"] == 1234.6
        assert result["errors"] == []
        assert result["success"] is True

    def test_index_project_uses_config_languages(self) -> None:
        """When languages=None, config.languages is used."""
        engine = _make_engine()
        engine.config.languages = ["typescript", "javascript"]
        engine.pipeline.index_project.return_value = IndexResult(
            files_indexed=0,
            nodes_created=0,
            edges_created=0,
            chunks_created=0,
            embeddings_created=0,
            duration_ms=10.0,
        )

        index_project(engine, "/project")

        call_kwargs = engine.pipeline.index_project.call_args
        assert call_kwargs[1]["languages"] == ["typescript", "javascript"]


# -------------------------------------------------------------------------
# TestUpdateProject
# -------------------------------------------------------------------------


class TestUpdateProject:
    def test_update_project_calls_pipeline(self) -> None:
        """Mock pipeline.update_project, verify call."""
        engine = _make_engine()
        engine.pipeline.update_project.return_value = IndexResult(
            files_indexed=2,
            nodes_created=8,
            edges_created=5,
            chunks_created=10,
            embeddings_created=10,
            duration_ms=500.0,
            errors=["some error"],
        )

        result = update_project(engine, "/my/project")

        engine.pipeline.update_project.assert_called_once_with(
            Path("/my/project"),
            languages=["python"],
            ignore_dirs={"__pycache__", ".git"},
        )
        assert result["files_indexed"] == 2
        assert result["nodes_created"] == 8
        assert result["duration_ms"] == 500.0
        assert result["errors"] == ["some error"]
        assert result["success"] is False


# -------------------------------------------------------------------------
# TestRememberRule
# -------------------------------------------------------------------------


class TestRememberRule:
    def test_remember_rule_calls_rules_manager(self) -> None:
        """Verify add_rule is called and result structure is correct."""
        engine = _make_engine()
        now = datetime.now(UTC)
        engine.rules.add_rule.return_value = RuleModel(
            id="rule-1",
            content="Always use type hints",
            scope="project",
            source="user",
            created_at=now,
        )

        result = remember_rule(engine, "Always use type hints", scope="project")

        engine.rules.add_rule.assert_called_once_with(
            "Always use type hints", scope="project", source="user"
        )
        assert result["id"] == "rule-1"
        assert result["content"] == "Always use type hints"
        assert result["scope"] == "project"
        assert result["source"] == "user"
        assert result["created_at"] == str(now)


# -------------------------------------------------------------------------
# TestRememberDecision
# -------------------------------------------------------------------------


class TestRememberDecision:
    def test_remember_decision_calls_decisions_manager(self) -> None:
        """Verify add_decision is called and result structure is correct."""
        engine = _make_engine()
        now = datetime.now(UTC)
        engine.decisions.add_decision.return_value = DecisionModel(
            id="dec-1",
            title="Use FastAPI",
            status="accepted",
            created_at=now,
        )

        result = remember_decision(engine, "Use FastAPI", status="accepted")

        engine.decisions.add_decision.assert_called_once_with(
            "Use FastAPI", status="accepted"
        )
        assert result["id"] == "dec-1"
        assert result["title"] == "Use FastAPI"
        assert result["status"] == "accepted"
        assert result["created_at"] == str(now)


# -------------------------------------------------------------------------
# TestGetMemory
# -------------------------------------------------------------------------


class TestGetMemory:
    def test_get_memory_returns_all(self) -> None:
        """Mock all three managers with data, verify combined result."""
        engine = _make_engine()
        engine.rules.get_rules.return_value = [
            RuleModel(id="r1", content="rule 1", scope="project"),
            RuleModel(id="r2", content="rule 2", scope="file"),
        ]
        engine.decisions.get_decisions.return_value = [
            DecisionModel(id="d1", title="decision 1", status="accepted"),
        ]
        engine.conventions.get_conventions.return_value = [
            ConventionModel(id="c1", pattern="snake_case", scope="project"),
        ]

        result = get_memory(engine)

        assert len(result["rules"]) == 2
        assert result["rules"][0]["id"] == "r1"
        assert result["rules"][0]["content"] == "rule 1"
        assert result["rules"][1]["scope"] == "file"

        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["title"] == "decision 1"

        assert len(result["conventions"]) == 1
        assert result["conventions"][0]["pattern"] == "snake_case"

        assert result["total"] == 4


# -------------------------------------------------------------------------
# TestGetSessionSummary
# -------------------------------------------------------------------------


class TestGetSessionSummary:
    def test_get_session_summary_returns_context(self) -> None:
        """Verify session methods are called and result is structured."""
        engine = _make_engine()
        engine.session.get_queries.return_value = ["q1", "q2"]
        engine.session.get_results.return_value = [
            {"query": "q1", "data": {"results": []}},
        ]
        engine.session.build_summary.return_value = "Queries:\n  - q1\n  - q2"

        result = get_session_summary(engine)

        assert result["queries"] == ["q1", "q2"]
        assert len(result["results"]) == 1
        assert result["summary"] == "Queries:\n  - q1\n  - q2"
        engine.session.get_queries.assert_called_once()
        engine.session.get_results.assert_called_once()
        engine.session.build_summary.assert_called_once()
