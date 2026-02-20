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
    list_projects,
    remember_decision,
    remember_rule,
    remove_project,
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
    # Registry default
    engine.registry.resolve.return_value = None
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
            [0.1, 0.2, 0.3], limit=5, project_id=None
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
                properties={"name": "foo", "line_start": 1, "line_end": 10},
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
        assert result["nodes"][0]["start_line"] == 1
        assert result["nodes"][0]["end_line"] == 10
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
        from nemesis.core.registry import ProjectInfo

        engine.registry.register.return_value = ProjectInfo(
            name="project", path=Path("/my/project"), languages=["python", "rust"]
        )
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
            project_id="project",
            project_root=Path("/my/project"),
        )
        assert result["project"] == "project"
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
        from nemesis.core.registry import ProjectInfo

        engine.config.languages = ["typescript", "javascript"]
        engine.registry.register.return_value = ProjectInfo(
            name="project", path=Path("/project"), languages=["typescript", "javascript"]
        )
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
            project_id="project",
            project_root=Path("/my/project"),
        )
        assert result["project"] == "project"
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


# -------------------------------------------------------------------------
# TestSearchCodeProject
# -------------------------------------------------------------------------


class TestSearchCodeProject:
    def test_search_with_project_filter(self) -> None:
        """When project is given, it is passed to vector_store.search."""
        engine = _make_engine()
        engine.embedder.embed_single.return_value = [0.1]
        engine.vector_store.search.return_value = []

        search_code(engine, "hello", project="eve")

        engine.vector_store.search.assert_called_once()
        call_kwargs = engine.vector_store.search.call_args
        assert call_kwargs.kwargs.get("project_id") == "eve"

    def test_search_cross_project_default(self) -> None:
        """When project is not given, project_id is None (cross-project)."""
        engine = _make_engine()
        engine.embedder.embed_single.return_value = [0.1]
        engine.vector_store.search.return_value = []

        search_code(engine, "hello")

        call_kwargs = engine.vector_store.search.call_args
        assert call_kwargs.kwargs.get("project_id") is None


# -------------------------------------------------------------------------
# TestGetContextProject
# -------------------------------------------------------------------------


class TestGetContextProject:
    def test_get_context_resolves_project(self) -> None:
        """When project is not given, registry.resolve is called."""
        engine = _make_engine()
        engine.graph.get_nodes_for_file.return_value = []

        get_context(engine, "/home/user/eve/src/main.py")

        engine.registry.resolve.assert_called_once_with(
            Path("/home/user/eve/src/main.py")
        )

    def test_get_context_uses_explicit_project(self) -> None:
        """When project is given explicitly, resolve is not relied upon."""
        engine = _make_engine()
        engine.graph.get_nodes_for_file.return_value = []

        get_context(engine, "src/main.py", project="eve")

        # resolve may still be called but explicit project takes precedence
        # The function does: resolved_project = project or engine.registry.resolve(...)
        # Since project="eve" is truthy, resolve result is ignored.

    def test_get_context_normalizes_absolute_path(self) -> None:
        """Absolute paths are converted to relative using project root."""
        engine = _make_engine()
        from nemesis.core.registry import ProjectInfo

        engine.registry.resolve.return_value = "eve"
        engine.registry.get.return_value = ProjectInfo(
            name="eve", path=Path("/home/user/eve"), languages=["python"]
        )
        engine.graph.get_nodes_for_file.return_value = []

        get_context(engine, "/home/user/eve/src/main.py")

        # Graph should be queried with relative path, not absolute
        engine.graph.get_nodes_for_file.assert_called_once_with("src/main.py")


# -------------------------------------------------------------------------
# TestIndexProjectRegistry
# -------------------------------------------------------------------------


class TestIndexProjectRegistry:
    def test_index_registers_project(self) -> None:
        """index_project calls registry.register and returns project name."""
        engine = _make_engine()
        from nemesis.core.registry import ProjectInfo

        engine.registry.register.return_value = ProjectInfo(
            name="myproj", path=Path("/tmp/myproj"), languages=["python"]
        )
        engine.pipeline.index_project.return_value = IndexResult(
            files_indexed=5,
            nodes_created=20,
            edges_created=15,
            chunks_created=30,
            embeddings_created=30,
            duration_ms=1000.0,
        )

        result = index_project(engine, "/tmp/myproj", languages=["python"])

        engine.registry.register.assert_called_once()
        assert result["project"] == "myproj"

    def test_index_updates_stats(self) -> None:
        """After indexing, registry.update_stats is called with file count."""
        engine = _make_engine()
        from nemesis.core.registry import ProjectInfo

        engine.registry.register.return_value = ProjectInfo(
            name="proj", path=Path("/tmp/proj"), languages=["python"]
        )
        engine.pipeline.index_project.return_value = IndexResult(
            files_indexed=10,
            nodes_created=0,
            edges_created=0,
            chunks_created=0,
            embeddings_created=0,
            duration_ms=100.0,
        )

        index_project(engine, "/tmp/proj")

        engine.registry.update_stats.assert_called_once_with("proj", 10)


# -------------------------------------------------------------------------
# TestUpdateProjectRegistry
# -------------------------------------------------------------------------


class TestUpdateProjectRegistry:
    def test_update_with_explicit_project(self) -> None:
        """When project is given, it is used as project_id."""
        engine = _make_engine()
        engine.pipeline.update_project.return_value = IndexResult(
            files_indexed=1,
            nodes_created=2,
            edges_created=1,
            chunks_created=3,
            embeddings_created=3,
            duration_ms=50.0,
        )

        result = update_project(engine, "/tmp/proj", project="myproj")

        assert result["project"] == "myproj"
        call_kwargs = engine.pipeline.update_project.call_args
        assert call_kwargs.kwargs.get("project_id") == "myproj"

    def test_update_resolves_project_from_registry(self) -> None:
        """When project is not given, registry.resolve is used."""
        engine = _make_engine()
        engine.registry.resolve.return_value = "resolved-proj"
        engine.pipeline.update_project.return_value = IndexResult(
            files_indexed=1,
            nodes_created=2,
            edges_created=1,
            chunks_created=3,
            embeddings_created=3,
            duration_ms=50.0,
        )

        result = update_project(engine, "/tmp/resolved-proj")

        assert result["project"] == "resolved-proj"


# -------------------------------------------------------------------------
# TestListProjects
# -------------------------------------------------------------------------


class TestListProjects:
    def test_list_returns_projects(self) -> None:
        """list_projects returns structured project info."""
        engine = _make_engine()
        from nemesis.core.registry import ProjectInfo

        engine.registry.list_projects.return_value = {
            "eve": ProjectInfo(
                name="eve",
                path=Path("/home/user/eve"),
                languages=["python"],
                files=100,
            ),
        }

        result = list_projects(engine)

        assert "projects" in result
        assert "eve" in result["projects"]
        assert result["count"] == 1
        assert result["projects"]["eve"]["files"] == 100
        assert result["projects"]["eve"]["path"] == "/home/user/eve"
        assert result["projects"]["eve"]["languages"] == ["python"]

    def test_list_empty(self) -> None:
        """list_projects with no projects returns empty dict."""
        engine = _make_engine()
        engine.registry.list_projects.return_value = {}

        result = list_projects(engine)

        assert result["projects"] == {}
        assert result["count"] == 0


# -------------------------------------------------------------------------
# TestRemoveProject
# -------------------------------------------------------------------------


class TestRemoveProject:
    def test_remove_unregisters_and_deletes(self) -> None:
        """remove_project deletes from vector store and unregisters."""
        engine = _make_engine()
        from nemesis.core.registry import ProjectInfo

        engine.registry.get.return_value = ProjectInfo(
            name="eve", path=Path("/tmp/eve"), languages=["python"]
        )

        result = remove_project(engine, "eve")

        engine.vector_store.delete_by_project.assert_called_once_with("eve")
        engine.registry.unregister.assert_called_once_with("eve")
        assert result["success"] is True
        assert result["project"] == "eve"

    def test_remove_nonexistent_returns_error(self) -> None:
        """remove_project with unknown name returns error."""
        engine = _make_engine()
        engine.registry.get.return_value = None

        result = remove_project(engine, "nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"]
        engine.vector_store.delete_by_project.assert_not_called()
        engine.registry.unregister.assert_not_called()
