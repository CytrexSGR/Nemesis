"""Tests for nemesis.memory.context.SessionContext."""

from __future__ import annotations

from nemesis.memory.context import SessionContext

# ──────────────────────────────────────────────────────────────────────
# test_empty_session
# ──────────────────────────────────────────────────────────────────────


class TestEmptySession:
    def test_empty_session(self) -> None:
        """Fresh session has empty queries and results."""
        ctx = SessionContext()
        assert ctx.get_queries() == []
        assert ctx.get_results() == []


# ──────────────────────────────────────────────────────────────────────
# test_add_query
# ──────────────────────────────────────────────────────────────────────


class TestAddQuery:
    def test_add_query(self) -> None:
        """Adds queries, get_queries returns them in order."""
        ctx = SessionContext()
        ctx.add_query("find all classes")
        ctx.add_query("show function signatures")

        queries = ctx.get_queries()
        assert len(queries) == 2
        assert queries[0] == "find all classes"
        assert queries[1] == "show function signatures"


# ──────────────────────────────────────────────────────────────────────
# test_add_result
# ──────────────────────────────────────────────────────────────────────


class TestAddResult:
    def test_add_result(self) -> None:
        """Adds result with query and data, get_results returns it."""
        ctx = SessionContext()
        ctx.add_result("find classes", {"files": ["a.py", "b.py"], "count": 2})

        results = ctx.get_results()
        assert len(results) == 1
        assert results[0]["query"] == "find classes"
        assert results[0]["data"]["files"] == ["a.py", "b.py"]
        assert results[0]["data"]["count"] == 2


# ──────────────────────────────────────────────────────────────────────
# test_build_summary
# ──────────────────────────────────────────────────────────────────────


class TestBuildSummary:
    def test_build_summary(self) -> None:
        """Builds text summary with query text and files from data."""
        ctx = SessionContext()
        ctx.add_query("find all classes")
        ctx.add_result("find all classes", {"files": ["models.py", "views.py"]})

        summary = ctx.build_summary()

        assert "Queries:" in summary
        assert "find all classes" in summary
        assert "Results:" in summary
        assert "models.py" in summary
        assert "views.py" in summary


# ──────────────────────────────────────────────────────────────────────
# test_clear
# ──────────────────────────────────────────────────────────────────────


class TestClear:
    def test_clear(self) -> None:
        """Clears all queries and results."""
        ctx = SessionContext()
        ctx.add_query("some query")
        ctx.add_result("some query", {"files": ["x.py"]})

        ctx.clear()

        assert ctx.get_queries() == []
        assert ctx.get_results() == []
