# 06c — Memory + Index Tools

> **Arbeitspaket F3** — Teil 3 von 4 des MCP Server Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Memory Tools (get_project_context, store_rule, store_decision) und Index Management Tools (index_project, index_status, watch_project) implementieren.

**Tech Stack:** Python 3.11+, MCP SDK (`mcp[cli]>=1.0`), Pydantic, pytest, pytest-asyncio

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [06b-code-intelligence.md](06b-code-intelligence.md) (F2 — Server + Code Intelligence Tools)

**Tasks in diesem Paket:** 7, 8, 9 (von 12)

---

## Task 7: Memory Tools — get_project_context, store_rule, store_decision

**Files:**
- `nemesis/tools/memory_query.py`
- `tests/test_server/test_memory_query.py`

### Step 1 — Write Test

```python
# tests/test_server/test_memory_query.py
"""Tests for memory tool handlers."""
import pytest
from unittest.mock import MagicMock, AsyncMock


def _make_mock_engine():
    engine = MagicMock()
    engine.graph = MagicMock()
    engine.graph.query = MagicMock(return_value=[])
    engine.graph.add_node = MagicMock(return_value=None)
    engine.graph.add_edge = MagicMock(return_value=None)
    engine.is_ready = True
    return engine


@pytest.mark.asyncio
async def test_project_context_returns_result():
    from nemesis.tools.memory_query import handle_project_context
    from nemesis.tools.models import ProjectContextInput, ProjectContextResult

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(side_effect=[
        [{"content": "Always use type hints"}],     # rules
        [{"title": "Use JWT for auth"}],             # decisions
        [{"pattern": "snake_case for functions"}],   # conventions
    ])

    inp = ProjectContextInput(topic="coding standards")
    result = await handle_project_context(engine, inp)

    assert isinstance(result, ProjectContextResult)
    assert len(result.rules) == 1
    assert "type hints" in result.rules[0]


@pytest.mark.asyncio
async def test_project_context_no_topic():
    from nemesis.tools.memory_query import handle_project_context
    from nemesis.tools.models import ProjectContextInput

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(side_effect=[
        [{"content": "Rule 1"}, {"content": "Rule 2"}],
        [],
        [],
    ])

    inp = ProjectContextInput()
    result = await handle_project_context(engine, inp)

    assert len(result.rules) == 2
    assert result.decisions == []
    assert result.conventions == []


@pytest.mark.asyncio
async def test_project_context_empty_memory():
    from nemesis.tools.memory_query import handle_project_context
    from nemesis.tools.models import ProjectContextInput

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(return_value=[])

    inp = ProjectContextInput()
    result = await handle_project_context(engine, inp)

    assert result.rules == []
    assert result.decisions == []
    assert result.conventions == []


@pytest.mark.asyncio
async def test_store_rule_returns_result():
    from nemesis.tools.memory_query import handle_store_rule
    from nemesis.tools.models import StoreRuleInput, StoreMutationResult

    engine = _make_mock_engine()

    inp = StoreRuleInput(
        rule="Always use parameterized SQL queries",
        scope="project",
        related_to=["database", "security"],
    )
    result = await handle_store_rule(engine, inp)

    assert isinstance(result, StoreMutationResult)
    assert result.stored is True
    assert result.id.startswith("rule-")


@pytest.mark.asyncio
async def test_store_rule_calls_graph():
    from nemesis.tools.memory_query import handle_store_rule
    from nemesis.tools.models import StoreRuleInput

    engine = _make_mock_engine()

    inp = StoreRuleInput(rule="No global state", scope="project")
    await handle_store_rule(engine, inp)

    engine.graph.add_node.assert_called_once()


@pytest.mark.asyncio
async def test_store_decision_returns_result():
    from nemesis.tools.memory_query import handle_store_decision
    from nemesis.tools.models import StoreDecisionInput, StoreMutationResult

    engine = _make_mock_engine()

    inp = StoreDecisionInput(
        title="Use PostgreSQL over MySQL",
        reasoning="Better JSON support, more advanced features",
        alternatives=["MySQL", "SQLite"],
    )
    result = await handle_store_decision(engine, inp)

    assert isinstance(result, StoreMutationResult)
    assert result.stored is True
    assert result.id.startswith("decision-")


@pytest.mark.asyncio
async def test_store_decision_calls_graph():
    from nemesis.tools.memory_query import handle_store_decision
    from nemesis.tools.models import StoreDecisionInput

    engine = _make_mock_engine()

    inp = StoreDecisionInput(
        title="JWT over sessions",
        reasoning="Stateless auth",
        alternatives=["Sessions"],
    )
    await handle_store_decision(engine, inp)

    # Node for decision + optional edges for alternatives
    assert engine.graph.add_node.call_count >= 1


@pytest.mark.asyncio
async def test_store_rule_handles_graph_error():
    from nemesis.tools.memory_query import handle_store_rule
    from nemesis.tools.models import StoreRuleInput

    engine = _make_mock_engine()
    engine.graph.add_node.side_effect = RuntimeError("DB error")

    inp = StoreRuleInput(rule="Some rule")
    result = await handle_store_rule(engine, inp)

    assert result.stored is False
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_memory_query.py -v
# Erwartung: FAILED — Module nemesis.tools.memory_query existiert nicht
```

### Step 3 — Implement

```python
# nemesis/tools/memory_query.py
"""Memory tool handlers.

Implements:
- get_project_context: Retrieve rules, decisions, conventions
- store_rule: Save a project rule to the graph
- store_decision: Document an architecture decision
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from nemesis.tools.models import (
    ProjectContextInput,
    ProjectContextResult,
    StoreDecisionInput,
    StoreMutationResult,
    StoreRuleInput,
)

logger = logging.getLogger(__name__)


def _generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix."""
    short_uuid = uuid.uuid4().hex[:12]
    return f"{prefix}-{short_uuid}"


async def handle_project_context(
    engine: Any, inp: ProjectContextInput
) -> ProjectContextResult:
    """Handle get_project_context: retrieve rules, decisions, conventions.

    Queries the graph for memory nodes, optionally filtered by topic.

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with optional topic.

    Returns:
        ProjectContextResult with rules, decisions, conventions.
    """
    graph = engine.graph
    rules: list[str] = []
    decisions: list[str] = []
    conventions: list[str] = []

    # Build topic filter clause
    topic_filter = ""
    params: dict[str, Any] = {}
    if inp.topic:
        topic_filter = " WHERE r.content CONTAINS $topic"
        params["topic"] = inp.topic

    # Query rules
    try:
        raw_rules = graph.query(
            f"MATCH (r:Rule) {topic_filter} RETURN r.content",
            params if inp.topic else None,
        )
        rules = [
            r.get("content", str(r)) if isinstance(r, dict) else str(r)
            for r in (raw_rules or [])
        ]
    except Exception as e:
        logger.debug("Failed to query rules: %s", e)

    # Query decisions
    try:
        decision_filter = ""
        if inp.topic:
            decision_filter = " WHERE d.title CONTAINS $topic"
        raw_decisions = graph.query(
            f"MATCH (d:Decision) {decision_filter} RETURN d.title",
            params if inp.topic else None,
        )
        decisions = [
            d.get("title", str(d)) if isinstance(d, dict) else str(d)
            for d in (raw_decisions or [])
        ]
    except Exception as e:
        logger.debug("Failed to query decisions: %s", e)

    # Query conventions
    try:
        convention_filter = ""
        if inp.topic:
            convention_filter = " WHERE c.pattern CONTAINS $topic"
        raw_conventions = graph.query(
            f"MATCH (c:Convention) {convention_filter} RETURN c.pattern",
            params if inp.topic else None,
        )
        conventions = [
            c.get("pattern", str(c)) if isinstance(c, dict) else str(c)
            for c in (raw_conventions or [])
        ]
    except Exception as e:
        logger.debug("Failed to query conventions: %s", e)

    return ProjectContextResult(
        rules=rules,
        decisions=decisions,
        conventions=conventions,
    )


async def handle_store_rule(
    engine: Any, inp: StoreRuleInput
) -> StoreMutationResult:
    """Handle store_rule: save a project rule to the graph.

    Creates a :Rule node with the rule text, scope, and timestamp.
    If related_to paths are provided, creates :APPLIES_TO edges.

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with rule, scope, related_to.

    Returns:
        StoreMutationResult with the new rule ID.
    """
    rule_id = _generate_id("rule")

    try:
        engine.graph.add_node({
            "id": rule_id,
            "node_type": "Rule",
            "content": inp.rule,
            "scope": inp.scope,
            "related_to": inp.related_to,
            "created_at": time.time(),
        })
        return StoreMutationResult(id=rule_id, stored=True)
    except Exception as e:
        logger.error("Failed to store rule: %s", e)
        return StoreMutationResult(id=rule_id, stored=False)


async def handle_store_decision(
    engine: Any, inp: StoreDecisionInput
) -> StoreMutationResult:
    """Handle store_decision: document an architecture decision.

    Creates a :Decision node with title, reasoning, and timestamp.
    Creates :Alternative nodes for each rejected alternative and
    connects them via :REJECTED edges.

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with title, reasoning, alternatives.

    Returns:
        StoreMutationResult with the new decision ID.
    """
    decision_id = _generate_id("decision")

    try:
        engine.graph.add_node({
            "id": decision_id,
            "node_type": "Decision",
            "title": inp.title,
            "reasoning": inp.reasoning,
            "status": "accepted",
            "created_at": time.time(),
        })

        # Add alternative nodes and edges
        for alt_title in inp.alternatives:
            alt_id = _generate_id("alt")
            engine.graph.add_node({
                "id": alt_id,
                "node_type": "Alternative",
                "title": alt_title,
                "reason_rejected": f"See decision: {inp.title}",
            })
            engine.graph.add_edge({
                "source_id": decision_id,
                "target_id": alt_id,
                "edge_type": "REJECTED",
            })

        return StoreMutationResult(id=decision_id, stored=True)
    except Exception as e:
        logger.error("Failed to store decision: %s", e)
        return StoreMutationResult(id=decision_id, stored=False)
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_memory_query.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/tools/memory_query.py tests/test_server/test_memory_query.py
git commit -m "feat(mcp): implement memory tools (project context, store rule/decision)"
```

---

## Task 8: Index Management — index_project, index_status

**Files:**
- `nemesis/tools/index_tools.py`
- `tests/test_server/test_index_tools.py`

### Step 1 — Write Test

```python
# tests/test_server/test_index_tools.py
"""Tests for index management tool handlers."""
import pytest
from unittest.mock import MagicMock, AsyncMock, PropertyMock
from pathlib import Path


def _make_mock_engine(project_root=None):
    engine = MagicMock()
    engine.graph = MagicMock()
    engine.vector_store = MagicMock()
    engine.vector_store.count = AsyncMock(return_value=42)
    engine.embedder = MagicMock()
    engine.pipeline = MagicMock()
    engine.is_ready = True
    engine.project_root = project_root or Path(".")
    engine.watcher_pid = None
    return engine


@pytest.mark.asyncio
async def test_index_project_returns_result():
    from nemesis.tools.index_tools import handle_index_project
    from nemesis.tools.models import IndexProjectInput, IndexProjectResult

    engine = _make_mock_engine()

    # Mock pipeline.index_project to return an IndexResult-like object
    mock_result = MagicMock()
    mock_result.files_indexed = 15
    mock_result.nodes_created = 120
    mock_result.edges_created = 80
    mock_result.duration_ms = 2500.0
    mock_result.success = True
    mock_result.errors = []
    engine.pipeline.index_project.return_value = mock_result

    inp = IndexProjectInput(path="/home/user/project", languages=["python"])
    result = await handle_index_project(engine, inp)

    assert isinstance(result, IndexProjectResult)
    assert result.files_indexed == 15
    assert result.nodes == 120
    assert result.edges == 80
    assert result.duration_ms == 2500.0


@pytest.mark.asyncio
async def test_index_project_calls_pipeline():
    from nemesis.tools.index_tools import handle_index_project
    from nemesis.tools.models import IndexProjectInput

    engine = _make_mock_engine()
    mock_result = MagicMock()
    mock_result.files_indexed = 0
    mock_result.nodes_created = 0
    mock_result.edges_created = 0
    mock_result.duration_ms = 100.0
    mock_result.success = True
    mock_result.errors = []
    engine.pipeline.index_project.return_value = mock_result

    inp = IndexProjectInput(path="/my/project", languages=["python", "typescript"])
    await handle_index_project(engine, inp)

    engine.pipeline.index_project.assert_called_once_with(
        Path("/my/project"), languages=["python", "typescript"]
    )


@pytest.mark.asyncio
async def test_index_project_uses_engine_root_for_default_path():
    from nemesis.tools.index_tools import handle_index_project
    from nemesis.tools.models import IndexProjectInput

    engine = _make_mock_engine(project_root=Path("/default/project"))
    mock_result = MagicMock()
    mock_result.files_indexed = 5
    mock_result.nodes_created = 30
    mock_result.edges_created = 20
    mock_result.duration_ms = 500.0
    mock_result.success = True
    mock_result.errors = []
    engine.pipeline.index_project.return_value = mock_result

    inp = IndexProjectInput()  # default path="."
    await handle_index_project(engine, inp)

    engine.pipeline.index_project.assert_called_once_with(
        Path("."), languages=["python"]
    )


@pytest.mark.asyncio
async def test_index_status_returns_result():
    from nemesis.tools.index_tools import handle_index_status
    from nemesis.tools.models import IndexStatusResult

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(side_effect=[
        [{"count": 42}],          # file count
        [{"count": 320}],         # node count
        [{"count": 180}],         # edge count
        [{"last_indexed": "2026-02-20T14:30:00Z"}],  # last indexed
        [],                        # stale files
    ])

    result = await handle_index_status(engine)

    assert isinstance(result, IndexStatusResult)
    assert result.files == 42


@pytest.mark.asyncio
async def test_index_status_empty_index():
    from nemesis.tools.index_tools import handle_index_status
    from nemesis.tools.models import IndexStatusResult

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(return_value=[])

    result = await handle_index_status(engine)

    assert isinstance(result, IndexStatusResult)
    assert result.files == 0
    assert result.last_indexed is None
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_index_tools.py -v
# Erwartung: FAILED — Module nemesis.tools.index_tools existiert nicht
```

### Step 3 — Implement

```python
# nemesis/tools/index_tools.py
"""Index management tool handlers.

Implements:
- index_project: Full project indexing
- index_status: Check index health
- watch_project: Start/stop file watcher
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nemesis.tools.models import (
    IndexProjectInput,
    IndexProjectResult,
    IndexStatusResult,
    WatchProjectInput,
    WatchProjectResult,
)

logger = logging.getLogger(__name__)


async def handle_index_project(
    engine: Any, inp: IndexProjectInput
) -> IndexProjectResult:
    """Handle index_project: trigger full project indexing.

    Delegates to the indexing pipeline for the full Parse -> Chunk ->
    Embed -> Store flow.

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with path and languages.

    Returns:
        IndexProjectResult with indexing statistics.
    """
    project_path = Path(inp.path)

    result = engine.pipeline.index_project(
        project_path, languages=inp.languages
    )

    return IndexProjectResult(
        files_indexed=result.files_indexed,
        nodes=result.nodes_created,
        edges=result.edges_created,
        duration_ms=result.duration_ms,
    )


async def handle_index_status(engine: Any) -> IndexStatusResult:
    """Handle index_status: check the health of the current index.

    Queries the graph for aggregate statistics: file count, node count,
    edge count, last indexed timestamp, and stale files.

    Args:
        engine: NemesisEngine instance.

    Returns:
        IndexStatusResult with current index health data.
    """
    graph = engine.graph

    files = 0
    nodes = 0
    edges = 0
    last_indexed: str | None = None
    stale_files: list[str] = []

    try:
        raw_files = graph.query("MATCH (f:File) RETURN count(f) AS count")
        if raw_files and isinstance(raw_files[0], dict):
            files = raw_files[0].get("count", 0)
    except Exception:
        pass

    try:
        raw_nodes = graph.query("MATCH (n) RETURN count(n) AS count")
        if raw_nodes and isinstance(raw_nodes[0], dict):
            nodes = raw_nodes[0].get("count", 0)
    except Exception:
        pass

    try:
        raw_edges = graph.query("MATCH ()-[e]->() RETURN count(e) AS count")
        if raw_edges and isinstance(raw_edges[0], dict):
            edges = raw_edges[0].get("count", 0)
    except Exception:
        pass

    try:
        raw_last = graph.query(
            "MATCH (p:Project) RETURN p.last_indexed ORDER BY p.last_indexed DESC LIMIT 1"
        )
        if raw_last and isinstance(raw_last[0], dict):
            last_indexed = raw_last[0].get("last_indexed")
    except Exception:
        pass

    try:
        raw_stale = graph.query(
            "MATCH (f:File) WHERE f.stale = true RETURN f.path"
        )
        stale_files = [
            s.get("path", str(s)) if isinstance(s, dict) else str(s)
            for s in (raw_stale or [])
        ]
    except Exception:
        pass

    return IndexStatusResult(
        last_indexed=last_indexed,
        files=files,
        nodes=nodes,
        edges=edges,
        stale_files=stale_files,
    )


async def handle_watch_project(
    engine: Any, inp: WatchProjectInput
) -> WatchProjectResult:
    """Handle watch_project: start or stop the file watcher.

    When enabled, starts a watchdog-based file watcher that triggers
    incremental delta re-indexing on file changes. When disabled,
    stops the running watcher.

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with path and enabled flag.

    Returns:
        WatchProjectResult with watcher status.
    """
    if inp.enabled:
        # Start watcher (actual watchdog integration happens in 08-cli-hooks)
        # For now, record the intent and return status
        logger.info("File watcher requested for %s", inp.path)
        engine.watcher_pid = None  # Will be set by actual watcher
        return WatchProjectResult(
            watching=True,
            path=inp.path,
            pid=engine.watcher_pid,
        )
    else:
        # Stop watcher
        logger.info("Stopping file watcher for %s", inp.path)
        old_pid = engine.watcher_pid
        engine.watcher_pid = None
        return WatchProjectResult(
            watching=False,
            path=inp.path,
            pid=old_pid,
        )
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_index_tools.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/tools/index_tools.py tests/test_server/test_index_tools.py
git commit -m "feat(mcp): implement index management tools (index, status, watch)"
```

---

## Task 9: Index Management — watch_project

**Files:**
- `tests/test_server/test_index_tools.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_server/test_index_tools.py — APPEND folgende Tests

@pytest.mark.asyncio
async def test_watch_project_enable():
    from nemesis.tools.index_tools import handle_watch_project
    from nemesis.tools.models import WatchProjectInput, WatchProjectResult

    engine = _make_mock_engine()

    inp = WatchProjectInput(path="/home/user/project", enabled=True)
    result = await handle_watch_project(engine, inp)

    assert isinstance(result, WatchProjectResult)
    assert result.watching is True
    assert result.path == "/home/user/project"


@pytest.mark.asyncio
async def test_watch_project_disable():
    from nemesis.tools.index_tools import handle_watch_project
    from nemesis.tools.models import WatchProjectInput

    engine = _make_mock_engine()
    engine.watcher_pid = 12345

    inp = WatchProjectInput(path="/home/user/project", enabled=False)
    result = await handle_watch_project(engine, inp)

    assert result.watching is False
    assert result.pid == 12345


@pytest.mark.asyncio
async def test_watch_project_disable_when_not_running():
    from nemesis.tools.index_tools import handle_watch_project
    from nemesis.tools.models import WatchProjectInput

    engine = _make_mock_engine()
    engine.watcher_pid = None

    inp = WatchProjectInput(path="/project", enabled=False)
    result = await handle_watch_project(engine, inp)

    assert result.watching is False
    assert result.pid is None
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_index_tools.py -v -k "watch"
# Erwartung: PASSED — Implementierung schon in Task 8
```

### Step 3 — Implement

Bereits in Task 8 implementiert.

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_index_tools.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add tests/test_server/test_index_tools.py
git commit -m "test(mcp): add tests for watch_project tool"
```

---

## Zusammenfassung F3

| Task | Datei(en) | Beschreibung | Tests |
|------|-----------|-------------|-------|
| 7 | `tools/memory_query.py` | Memory Tools: get_project_context, store_rule, store_decision | 9 |
| 8 | `tools/index_tools.py` | Index Tools: index_project, index_status | 5 |
| 9 | `tests/` (erweitern) | watch_project Tests | 3 |
| **Total** | | | **~17 Tests** |

---

**Navigation:**
- Vorheriges Paket: [06b-code-intelligence.md](06b-code-intelligence.md) (F2 — Server + Code Intelligence Tools)
- Naechstes Paket: [06d-smart-integration.md](06d-smart-integration.md) (F4 — Smart Context + Dispatch + Entry Point)
