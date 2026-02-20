# 06b — Server + Code Intelligence Tools

> **Arbeitspaket F2** — Teil 2 von 4 des MCP Server Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** MCP Server Skeleton mit stdio Transport aufsetzen und alle Code Intelligence Tool-Handler implementieren (get_code_semantics, get_dependencies, get_architecture, get_impact).

**Tech Stack:** Python 3.11+, MCP SDK (`mcp[cli]>=1.0`), Pydantic, pytest, pytest-asyncio

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [06a-server-foundation.md](06a-server-foundation.md) (F1 — Pydantic Models + NemesisEngine)

**Tasks in diesem Paket:** 3, 4, 5, 6 (von 12)

---

## Task 3: MCP Server Skeleton mit stdio Transport

**Files:**
- `nemesis/core/server.py`
- `tests/test_server/test_server.py`

### Step 1 — Write Test

```python
# tests/test_server/test_server.py
"""Tests for MCP server setup and tool registration."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


def test_server_module_importable():
    from nemesis.core import server
    assert server is not None


def test_create_server_returns_server():
    from nemesis.core.server import create_server
    from mcp.server import Server

    mock_engine = MagicMock()
    mock_engine.is_ready = True

    srv = create_server(mock_engine)
    assert isinstance(srv, Server)


def test_create_server_has_name():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True

    srv = create_server(mock_engine)
    assert srv.name == "nemesis"


@pytest.mark.asyncio
async def test_server_lists_all_tools():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True

    srv = create_server(mock_engine)

    # Use the server's list_tools handler
    from mcp.types import ListToolsRequest
    tools = await srv.list_tools()

    tool_names = [t.name for t in tools]
    expected_tools = [
        "get_code_semantics",
        "get_dependencies",
        "get_architecture",
        "get_impact",
        "get_project_context",
        "store_rule",
        "store_decision",
        "index_project",
        "index_status",
        "watch_project",
        "get_smart_context",
    ]
    for name in expected_tools:
        assert name in tool_names, f"Tool {name} not registered"


@pytest.mark.asyncio
async def test_server_tool_count():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True

    srv = create_server(mock_engine)
    tools = await srv.list_tools()
    assert len(tools) == 11


@pytest.mark.asyncio
async def test_each_tool_has_description():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True

    srv = create_server(mock_engine)
    tools = await srv.list_tools()

    for tool in tools:
        assert tool.description is not None, f"Tool {tool.name} missing description"
        assert len(tool.description) > 10, f"Tool {tool.name} description too short"


@pytest.mark.asyncio
async def test_each_tool_has_input_schema():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True

    srv = create_server(mock_engine)
    tools = await srv.list_tools()

    for tool in tools:
        assert tool.inputSchema is not None, f"Tool {tool.name} missing inputSchema"
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_server.py -v
# Erwartung: FAILED — Module nemesis.core.server oder create_server existiert nicht
```

### Step 3 — Implement

```python
# nemesis/core/server.py
"""Nemesis MCP Server — exposes all tools via stdio transport.

This is the main entry point for Claude Code integration. The server
registers 11 tools across three categories:
- Code Intelligence (4 tools)
- Memory (3 tools)
- Index Management (3 tools)
- Smart Context (1 tool)

Usage:
    nemesis serve   # starts the MCP server on stdio
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from nemesis.tools.models import (
    ArchitectureInput,
    ArchitectureResult,
    CodeSemanticsInput,
    CodeSemanticsResult,
    DependenciesInput,
    DependenciesResult,
    ImpactInput,
    ImpactResult,
    IndexProjectInput,
    IndexProjectResult,
    IndexStatusResult,
    ProjectContextInput,
    ProjectContextResult,
    SmartContextInput,
    SmartContextResult,
    StoreDecisionInput,
    StoreMutationResult,
    StoreRuleInput,
    WatchProjectInput,
    WatchProjectResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions — static metadata for tool registration
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_code_semantics",
        "description": (
            "Natural language code search. Finds functions, classes, and code "
            "snippets semantically related to the query using vector search "
            "and graph expansion."
        ),
        "model": CodeSemanticsInput,
    },
    {
        "name": "get_dependencies",
        "description": (
            "Dependency traversal for a symbol. Returns incoming dependencies, "
            "outgoing dependencies, and call chains up to the specified depth."
        ),
        "model": DependenciesInput,
    },
    {
        "name": "get_architecture",
        "description": (
            "Architecture overview. Returns modules, key classes, data flow, "
            "and entry points for the given scope (project, module, or file)."
        ),
        "model": ArchitectureInput,
    },
    {
        "name": "get_impact",
        "description": (
            "Change impact analysis. Given a file and optionally a function, "
            "returns direct dependents, transitive dependents, and test coverage."
        ),
        "model": ImpactInput,
    },
    {
        "name": "get_project_context",
        "description": (
            "Retrieve project rules, architecture decisions, and coding "
            "conventions. Optionally filter by topic."
        ),
        "model": ProjectContextInput,
    },
    {
        "name": "store_rule",
        "description": (
            "Save a project rule to the memory layer. Rules are automatically "
            "included in relevant context packages."
        ),
        "model": StoreRuleInput,
    },
    {
        "name": "store_decision",
        "description": (
            "Document an architecture decision with reasoning and alternatives "
            "considered. Stored in the knowledge graph for future context."
        ),
        "model": StoreDecisionInput,
    },
    {
        "name": "index_project",
        "description": (
            "Trigger a full project index. Parses all code files, builds the "
            "knowledge graph, and generates vector embeddings."
        ),
        "model": IndexProjectInput,
    },
    {
        "name": "index_status",
        "description": (
            "Check the health of the current index. Returns file count, node "
            "count, edge count, last indexed time, and stale files."
        ),
        "model": None,
    },
    {
        "name": "watch_project",
        "description": (
            "Start or stop the file watcher for incremental index updates. "
            "When enabled, file changes trigger automatic delta re-indexing."
        ),
        "model": WatchProjectInput,
    },
    {
        "name": "get_smart_context",
        "description": (
            "The CLAUDE.md replacement. Given a task description and token budget, "
            "returns a curated context package combining relevant code, rules, "
            "decisions, and architecture — sized to fit the token budget."
        ),
        "model": SmartContextInput,
    },
]


def create_server(engine: Any) -> Server:
    """Create and configure the Nemesis MCP server.

    Registers all 11 tools and their handlers. The engine is captured
    in the handler closures so each tool has access to all backends.

    Args:
        engine: NemesisEngine instance with all backends wired.

    Returns:
        Configured MCP Server ready to run.
    """
    server = Server("nemesis")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return all available Nemesis tools."""
        tools = []
        for defn in _TOOL_DEFINITIONS:
            if defn["model"] is not None:
                schema = defn["model"].model_json_schema()
            else:
                schema = {"type": "object", "properties": {}}
            tools.append(
                Tool(
                    name=defn["name"],
                    description=defn["description"],
                    inputSchema=schema,
                )
            )
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Dispatch a tool call to the appropriate handler."""
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        try:
            result = await handler(engine, arguments)
            if hasattr(result, "model_dump_json"):
                text = result.model_dump_json(indent=2)
            else:
                text = json.dumps(result, indent=2, default=str)
            return [TextContent(type="text", text=text)]
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e, exc_info=True)
            return [TextContent(type="text", text=f"Error: {e}")]

    return server


# ---------------------------------------------------------------------------
# Tool handlers — imported from tool modules
# ---------------------------------------------------------------------------

async def _handle_code_semantics(engine: Any, arguments: dict) -> CodeSemanticsResult:
    from nemesis.tools.code_query import handle_code_semantics
    return await handle_code_semantics(engine, CodeSemanticsInput(**arguments))


async def _handle_dependencies(engine: Any, arguments: dict) -> DependenciesResult:
    from nemesis.tools.code_query import handle_dependencies
    return await handle_dependencies(engine, DependenciesInput(**arguments))


async def _handle_architecture(engine: Any, arguments: dict) -> ArchitectureResult:
    from nemesis.tools.code_query import handle_architecture
    return await handle_architecture(engine, ArchitectureInput(**arguments))


async def _handle_impact(engine: Any, arguments: dict) -> ImpactResult:
    from nemesis.tools.code_query import handle_impact
    return await handle_impact(engine, ImpactInput(**arguments))


async def _handle_project_context(engine: Any, arguments: dict) -> ProjectContextResult:
    from nemesis.tools.memory_query import handle_project_context
    return await handle_project_context(engine, ProjectContextInput(**arguments))


async def _handle_store_rule(engine: Any, arguments: dict) -> StoreMutationResult:
    from nemesis.tools.memory_query import handle_store_rule
    return await handle_store_rule(engine, StoreRuleInput(**arguments))


async def _handle_store_decision(engine: Any, arguments: dict) -> StoreMutationResult:
    from nemesis.tools.memory_query import handle_store_decision
    return await handle_store_decision(engine, StoreDecisionInput(**arguments))


async def _handle_index_project(engine: Any, arguments: dict) -> IndexProjectResult:
    from nemesis.tools.index_tools import handle_index_project
    return await handle_index_project(engine, IndexProjectInput(**arguments))


async def _handle_index_status(engine: Any, arguments: dict) -> IndexStatusResult:
    from nemesis.tools.index_tools import handle_index_status
    return await handle_index_status(engine)


async def _handle_watch_project(engine: Any, arguments: dict) -> WatchProjectResult:
    from nemesis.tools.index_tools import handle_watch_project
    return await handle_watch_project(engine, WatchProjectInput(**arguments))


async def _handle_smart_context(engine: Any, arguments: dict) -> SmartContextResult:
    from nemesis.tools.code_query import handle_smart_context
    return await handle_smart_context(engine, SmartContextInput(**arguments))


_TOOL_HANDLERS: dict = {
    "get_code_semantics": _handle_code_semantics,
    "get_dependencies": _handle_dependencies,
    "get_architecture": _handle_architecture,
    "get_impact": _handle_impact,
    "get_project_context": _handle_project_context,
    "store_rule": _handle_store_rule,
    "store_decision": _handle_store_decision,
    "index_project": _handle_index_project,
    "index_status": _handle_index_status,
    "watch_project": _handle_watch_project,
    "get_smart_context": _handle_smart_context,
}


async def run_server(engine: Any) -> None:
    """Run the MCP server on stdio transport.

    This is the main entry point called by `nemesis serve`.

    Args:
        engine: NemesisEngine instance.
    """
    from mcp.server.stdio import stdio_server

    server = create_server(engine)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_server.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/core/server.py tests/test_server/test_server.py
git commit -m "feat(mcp): add MCP server skeleton with all 11 tools registered"
```

---

## Task 4: Code Intelligence — get_code_semantics

**Files:**
- `nemesis/tools/code_query.py`
- `tests/test_server/test_code_query.py`

### Step 1 — Write Test

```python
# tests/test_server/test_code_query.py
"""Tests for code intelligence tool handlers."""
import pytest
from unittest.mock import MagicMock, AsyncMock


def _make_mock_engine():
    """Create a mock NemesisEngine with all backends."""
    from unittest.mock import MagicMock, AsyncMock

    engine = MagicMock()
    engine.graph = MagicMock()
    engine.vector_store = MagicMock()
    engine.embedder = MagicMock()
    engine.pipeline = MagicMock()
    engine.project_root = MagicMock()
    engine.is_ready = True

    # Default: embedding returns a vector
    engine.embedder.embed_single = AsyncMock(return_value=[0.1] * 384)

    # Default: vector search returns results
    engine.vector_store.search = AsyncMock(return_value=[])

    # Default: graph queries return empty lists
    engine.graph.query = MagicMock(return_value=[])
    engine.graph.get_dependents = MagicMock(return_value=[])
    engine.graph.get_dependencies_of = MagicMock(return_value=[])

    return engine


@pytest.mark.asyncio
async def test_code_semantics_returns_result():
    from nemesis.tools.code_query import handle_code_semantics
    from nemesis.tools.models import CodeSemanticsInput, CodeSemanticsResult

    engine = _make_mock_engine()
    engine.vector_store.search = AsyncMock(return_value=[])

    inp = CodeSemanticsInput(query="authentication flow", limit=5)
    result = await handle_code_semantics(engine, inp)

    assert isinstance(result, CodeSemanticsResult)


@pytest.mark.asyncio
async def test_code_semantics_calls_embed_and_search():
    from nemesis.tools.code_query import handle_code_semantics
    from nemesis.tools.models import CodeSemanticsInput

    engine = _make_mock_engine()

    inp = CodeSemanticsInput(query="auth flow", limit=5)
    await handle_code_semantics(engine, inp)

    engine.embedder.embed_single.assert_awaited_once_with("auth flow")
    engine.vector_store.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_code_semantics_with_vector_results():
    from nemesis.tools.code_query import handle_code_semantics
    from nemesis.tools.models import CodeSemanticsInput

    engine = _make_mock_engine()

    mock_result = MagicMock()
    mock_result.id = "func-001:chunk-0"
    mock_result.text = "def authenticate(user, pw):\n    ..."
    mock_result.score = 0.92
    mock_result.metadata = {
        "file_path": "src/auth.py",
        "parent_node_id": "func-001",
        "parent_type": "Function",
        "start_line": 10,
        "end_line": 20,
    }

    engine.vector_store.search = AsyncMock(return_value=[mock_result])
    engine.graph.get_related_files = MagicMock(return_value=["src/utils.py"])

    inp = CodeSemanticsInput(query="authentication", limit=5)
    result = await handle_code_semantics(engine, inp)

    assert len(result.matches) == 1
    assert result.matches[0].file == "src/auth.py"
    assert result.matches[0].relevance == pytest.approx(0.92)
    assert result.matches[0].snippet == "def authenticate(user, pw):\n    ..."


@pytest.mark.asyncio
async def test_code_semantics_respects_limit():
    from nemesis.tools.code_query import handle_code_semantics
    from nemesis.tools.models import CodeSemanticsInput

    engine = _make_mock_engine()

    inp = CodeSemanticsInput(query="auth", limit=3)
    await handle_code_semantics(engine, inp)

    call_kwargs = engine.vector_store.search.call_args
    assert call_kwargs[1]["limit"] == 3 or call_kwargs.kwargs.get("limit") == 3
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_code_query.py -v -k "code_semantics"
# Erwartung: FAILED — Module nemesis.tools.code_query existiert nicht
```

### Step 3 — Implement

```python
# nemesis/tools/code_query.py
"""Code intelligence tool handlers.

Implements:
- get_code_semantics: Natural language code search (vector + graph)
- get_dependencies: Dependency traversal
- get_architecture: Architecture overview
- get_impact: Change impact analysis
- get_smart_context: Curated context package (CLAUDE.md replacement)
"""
from __future__ import annotations

import logging
from typing import Any

from nemesis.tools.models import (
    ArchitectureInput,
    ArchitectureResult,
    CodeMatch,
    CodeSemanticsInput,
    CodeSemanticsResult,
    DependenciesInput,
    DependenciesResult,
    DependencyNode,
    ImpactInput,
    ImpactResult,
    SmartContextInput,
    SmartContextResult,
)

logger = logging.getLogger(__name__)


async def handle_code_semantics(
    engine: Any, inp: CodeSemanticsInput
) -> CodeSemanticsResult:
    """Handle get_code_semantics: natural language code search.

    Pipeline:
    1. Embed the query text
    2. Vector similarity search in LanceDB
    3. For each match, look up related files via graph
    4. Return ranked matches

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with query and limit.

    Returns:
        CodeSemanticsResult with ranked code matches.
    """
    # 1. Embed query
    query_vector = await engine.embedder.embed_single(inp.query)

    # 2. Vector search
    raw_results = await engine.vector_store.search(
        query_embedding=query_vector, limit=inp.limit
    )

    # 3. Build matches with graph expansion
    matches: list[CodeMatch] = []
    for hit in raw_results:
        metadata = hit.metadata if hasattr(hit, "metadata") else {}
        file_path = metadata.get("file_path", "unknown")
        parent_type = metadata.get("parent_type", "unknown")
        parent_id = metadata.get("parent_node_id", "")

        # Try to get related files from graph
        related_files: list[str] = []
        try:
            related_files = engine.graph.get_related_files(file_path)
        except Exception:
            pass

        # Extract function name from parent_id
        function_name = parent_id.split(":")[0] if parent_id else parent_type

        matches.append(
            CodeMatch(
                file=file_path,
                function=function_name,
                snippet=hit.text,
                related_files=related_files,
                relevance=round(max(0.0, min(1.0, hit.score)), 4),
            )
        )

    return CodeSemanticsResult(matches=matches)


async def handle_dependencies(
    engine: Any, inp: DependenciesInput
) -> DependenciesResult:
    """Handle get_dependencies: dependency traversal for a symbol.

    Uses graph traversal to find incoming and outgoing dependencies
    up to the specified depth.

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with symbol, depth, direction.

    Returns:
        DependenciesResult with the dependency tree.
    """
    graph = engine.graph

    # Find the root node
    nodes = graph.query(
        "MATCH (n) WHERE n.name = $name RETURN n.name, n.node_type, n.file LIMIT 1",
        {"name": inp.symbol},
    )

    if not nodes:
        return DependenciesResult(
            root=DependencyNode(
                name=inp.symbol,
                node_type="unknown",
                file="not found",
                deps_in=[],
                deps_out=[],
                call_chain=[],
            )
        )

    node_data = nodes[0]
    node_name = node_data.get("name", inp.symbol) if isinstance(node_data, dict) else inp.symbol
    node_type = node_data.get("node_type", "unknown") if isinstance(node_data, dict) else "unknown"
    node_file = node_data.get("file", "unknown") if isinstance(node_data, dict) else "unknown"

    deps_in: list[str] = []
    deps_out: list[str] = []
    call_chain: list[str] = []

    if inp.direction in ("in", "both"):
        try:
            deps_in = [
                d.get("name", str(d)) if isinstance(d, dict) else str(d)
                for d in graph.get_dependents(inp.symbol, depth=inp.depth)
            ]
        except Exception:
            pass

    if inp.direction in ("out", "both"):
        try:
            deps_out = [
                d.get("name", str(d)) if isinstance(d, dict) else str(d)
                for d in graph.get_dependencies_of(inp.symbol, depth=inp.depth)
            ]
        except Exception:
            pass

    # Build call chain representation
    for dep in deps_in:
        call_chain.append(f"{dep} -> {inp.symbol}")
    for dep in deps_out:
        call_chain.append(f"{inp.symbol} -> {dep}")

    return DependenciesResult(
        root=DependencyNode(
            name=node_name,
            node_type=node_type,
            file=node_file,
            deps_in=deps_in,
            deps_out=deps_out,
            call_chain=call_chain,
        )
    )


async def handle_architecture(
    engine: Any, inp: ArchitectureInput
) -> ArchitectureResult:
    """Handle get_architecture: architecture overview.

    Aggregates graph data to produce a high-level overview of
    modules, key classes, data flow, and entry points.

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with scope.

    Returns:
        ArchitectureResult with architecture summary.
    """
    graph = engine.graph

    modules: list[str] = []
    key_classes: list[str] = []
    data_flow: list[str] = []
    entry_points: list[str] = []

    try:
        # Get modules/files
        raw_modules = graph.query(
            "MATCH (f:File) RETURN f.path ORDER BY f.path"
        )
        modules = [
            m.get("path", str(m)) if isinstance(m, dict) else str(m)
            for m in (raw_modules or [])
        ]
    except Exception:
        pass

    try:
        # Get key classes
        raw_classes = graph.query(
            "MATCH (c:Class) RETURN c.name, c.file ORDER BY c.name"
        )
        key_classes = [
            c.get("name", str(c)) if isinstance(c, dict) else str(c)
            for c in (raw_classes or [])
        ]
    except Exception:
        pass

    try:
        # Get imports (data flow)
        raw_imports = graph.query(
            "MATCH (a:File)-[:IMPORTS]->(b:File) "
            "RETURN a.path AS src, b.path AS dst"
        )
        data_flow = [
            f"{i['src']} -> {i['dst']}" if isinstance(i, dict) else str(i)
            for i in (raw_imports or [])
        ]
    except Exception:
        pass

    try:
        # Get entry points (files with a main function or __main__)
        raw_entries = graph.query(
            "MATCH (f:File)-[:CONTAINS]->(fn:Function) "
            "WHERE fn.name IN ['main', '__main__'] "
            "RETURN f.path, fn.name"
        )
        entry_points = [
            f"{e['path']}:{e['name']}" if isinstance(e, dict) else str(e)
            for e in (raw_entries or [])
        ]
    except Exception:
        pass

    return ArchitectureResult(
        modules=modules,
        key_classes=key_classes,
        data_flow=data_flow,
        entry_points=entry_points,
    )


async def handle_impact(engine: Any, inp: ImpactInput) -> ImpactResult:
    """Handle get_impact: change impact analysis.

    Given a file (and optionally a function), find what depends on it
    directly and transitively, plus related test files.

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with file and optional function.

    Returns:
        ImpactResult with dependents and test coverage.
    """
    graph = engine.graph

    direct: list[str] = []
    transitive: list[str] = []
    tests: list[str] = []

    try:
        # Direct dependents: files that import this file
        raw_direct = graph.query(
            "MATCH (other:File)-[:IMPORTS]->(target:File) "
            "WHERE target.path = $path "
            "RETURN other.path",
            {"path": inp.file},
        )
        direct = [
            d.get("path", str(d)) if isinstance(d, dict) else str(d)
            for d in (raw_direct or [])
        ]
    except Exception:
        pass

    try:
        # Transitive dependents (depth 2)
        raw_transitive = graph.query(
            "MATCH (other:File)-[:IMPORTS*2]->(target:File) "
            "WHERE target.path = $path "
            "RETURN DISTINCT other.path",
            {"path": inp.file},
        )
        transitive = [
            d.get("path", str(d)) if isinstance(d, dict) else str(d)
            for d in (raw_transitive or [])
        ]
    except Exception:
        pass

    # Test files: any file with 'test' in the path that depends on this file
    tests = [f for f in direct + transitive if "test" in f.lower()]

    return ImpactResult(
        direct_dependents=direct,
        transitive_dependents=transitive,
        test_coverage=tests,
    )


async def handle_smart_context(
    engine: Any, inp: SmartContextInput
) -> SmartContextResult:
    """Handle get_smart_context: the CLAUDE.md replacement.

    Combines vector search, graph traversal, and memory retrieval
    to produce a curated context package sized to the token budget.

    Pipeline:
    1. Embed the task description
    2. Vector search for relevant code
    3. Graph expansion for dependencies
    4. Memory retrieval for rules + decisions
    5. Architecture overview
    6. Rank and trim to token budget

    Args:
        engine: NemesisEngine instance.
        inp: Validated input with task and max_tokens.

    Returns:
        SmartContextResult with curated context.
    """
    from nemesis.tools.models import CodeSemanticsInput

    # 1. Find relevant code via vector search
    code_result = await handle_code_semantics(
        engine, CodeSemanticsInput(query=inp.task, limit=10)
    )

    code_context: list[dict] = []
    token_count = 0

    for match in code_result.matches:
        snippet_tokens = len(match.snippet) // 4  # Rough estimate
        if token_count + snippet_tokens > inp.max_tokens * 0.6:
            break
        code_context.append({
            "file": match.file,
            "function": match.function,
            "snippet": match.snippet,
            "relevance": match.relevance,
        })
        token_count += snippet_tokens

    # 2. Get rules and decisions from memory
    rules: list[str] = []
    decisions: list[str] = []
    try:
        from nemesis.tools.memory_query import handle_project_context
        from nemesis.tools.models import ProjectContextInput

        memory = await handle_project_context(
            engine, ProjectContextInput(topic=inp.task)
        )
        rules = memory.rules
        decisions = memory.decisions
    except Exception:
        pass

    # 3. Get architecture overview
    architecture = ""
    try:
        arch_result = await handle_architecture(
            engine, ArchitectureInput(scope="project")
        )
        parts = []
        if arch_result.modules:
            parts.append(f"Modules: {', '.join(arch_result.modules[:10])}")
        if arch_result.key_classes:
            parts.append(f"Key classes: {', '.join(arch_result.key_classes[:10])}")
        if arch_result.entry_points:
            parts.append(f"Entry points: {', '.join(arch_result.entry_points[:5])}")
        architecture = "; ".join(parts)
    except Exception:
        pass

    # Estimate total token count
    for rule in rules:
        token_count += len(rule) // 4
    for dec in decisions:
        token_count += len(dec) // 4
    token_count += len(architecture) // 4

    return SmartContextResult(
        code_context=code_context,
        rules=rules,
        decisions=decisions,
        architecture=architecture,
        token_count=token_count,
    )
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_code_query.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/tools/code_query.py tests/test_server/test_code_query.py
git commit -m "feat(mcp): implement code intelligence tools (semantics, deps, arch, impact, smart context)"
```

---

## Task 5: Code Intelligence — get_dependencies

**Files:**
- `tests/test_server/test_code_query.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_server/test_code_query.py — APPEND folgende Tests

@pytest.mark.asyncio
async def test_dependencies_returns_result():
    from nemesis.tools.code_query import handle_dependencies
    from nemesis.tools.models import DependenciesInput, DependenciesResult

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(return_value=[
        {"name": "UserService", "node_type": "Class", "file": "src/user.py"},
    ])
    engine.graph.get_dependents = MagicMock(return_value=[
        {"name": "AuthService"},
    ])
    engine.graph.get_dependencies_of = MagicMock(return_value=[
        {"name": "Database"},
    ])

    inp = DependenciesInput(symbol="UserService", depth=2, direction="both")
    result = await handle_dependencies(engine, inp)

    assert isinstance(result, DependenciesResult)
    assert result.root.name == "UserService"
    assert "AuthService" in result.root.deps_in
    assert "Database" in result.root.deps_out


@pytest.mark.asyncio
async def test_dependencies_not_found():
    from nemesis.tools.code_query import handle_dependencies
    from nemesis.tools.models import DependenciesInput

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(return_value=[])

    inp = DependenciesInput(symbol="NonexistentClass")
    result = await handle_dependencies(engine, inp)

    assert result.root.node_type == "unknown"
    assert result.root.file == "not found"


@pytest.mark.asyncio
async def test_dependencies_direction_in_only():
    from nemesis.tools.code_query import handle_dependencies
    from nemesis.tools.models import DependenciesInput

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(return_value=[
        {"name": "Foo", "node_type": "Class", "file": "foo.py"},
    ])
    engine.graph.get_dependents = MagicMock(return_value=[{"name": "Bar"}])
    engine.graph.get_dependencies_of = MagicMock(return_value=[{"name": "Baz"}])

    inp = DependenciesInput(symbol="Foo", direction="in")
    result = await handle_dependencies(engine, inp)

    assert "Bar" in result.root.deps_in
    assert result.root.deps_out == []


@pytest.mark.asyncio
async def test_dependencies_direction_out_only():
    from nemesis.tools.code_query import handle_dependencies
    from nemesis.tools.models import DependenciesInput

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(return_value=[
        {"name": "Foo", "node_type": "Class", "file": "foo.py"},
    ])
    engine.graph.get_dependents = MagicMock(return_value=[{"name": "Bar"}])
    engine.graph.get_dependencies_of = MagicMock(return_value=[{"name": "Baz"}])

    inp = DependenciesInput(symbol="Foo", direction="out")
    result = await handle_dependencies(engine, inp)

    assert result.root.deps_in == []
    assert "Baz" in result.root.deps_out
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_code_query.py -v -k "dependencies"
# Erwartung: PASSED — Implementierung schon in Task 4
```

### Step 3 — Implement

Bereits in Task 4 implementiert.

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_code_query.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add tests/test_server/test_code_query.py
git commit -m "test(mcp): add tests for get_dependencies tool"
```

---

## Task 6: Code Intelligence — get_architecture und get_impact

**Files:**
- `tests/test_server/test_code_query.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_server/test_code_query.py — APPEND folgende Tests

@pytest.mark.asyncio
async def test_architecture_returns_result():
    from nemesis.tools.code_query import handle_architecture
    from nemesis.tools.models import ArchitectureInput, ArchitectureResult

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(side_effect=[
        [{"path": "src/auth.py"}, {"path": "src/api.py"}],  # files
        [{"name": "AuthService"}, {"name": "UserRepo"}],     # classes
        [{"src": "src/api.py", "dst": "src/auth.py"}],       # imports
        [{"path": "src/main.py", "name": "main"}],           # entries
    ])

    inp = ArchitectureInput(scope="project")
    result = await handle_architecture(engine, inp)

    assert isinstance(result, ArchitectureResult)
    assert "src/auth.py" in result.modules
    assert "AuthService" in result.key_classes
    assert len(result.data_flow) >= 1
    assert len(result.entry_points) >= 1


@pytest.mark.asyncio
async def test_architecture_empty_graph():
    from nemesis.tools.code_query import handle_architecture
    from nemesis.tools.models import ArchitectureInput

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(return_value=[])

    inp = ArchitectureInput(scope="project")
    result = await handle_architecture(engine, inp)

    assert result.modules == []
    assert result.key_classes == []


@pytest.mark.asyncio
async def test_impact_returns_result():
    from nemesis.tools.code_query import handle_impact
    from nemesis.tools.models import ImpactInput, ImpactResult

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(side_effect=[
        [{"path": "src/api.py"}, {"path": "tests/test_auth.py"}],  # direct
        [{"path": "src/main.py"}],  # transitive
    ])

    inp = ImpactInput(file="src/auth.py", function="login")
    result = await handle_impact(engine, inp)

    assert isinstance(result, ImpactResult)
    assert "src/api.py" in result.direct_dependents
    assert "tests/test_auth.py" in result.test_coverage


@pytest.mark.asyncio
async def test_impact_no_dependents():
    from nemesis.tools.code_query import handle_impact
    from nemesis.tools.models import ImpactInput

    engine = _make_mock_engine()
    engine.graph.query = MagicMock(return_value=[])

    inp = ImpactInput(file="src/isolated.py")
    result = await handle_impact(engine, inp)

    assert result.direct_dependents == []
    assert result.transitive_dependents == []
    assert result.test_coverage == []
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_code_query.py -v -k "architecture or impact"
# Erwartung: PASSED — Implementierung schon in Task 4
```

### Step 3 — Implement

Bereits in Task 4 implementiert.

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_code_query.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add tests/test_server/test_code_query.py
git commit -m "test(mcp): add tests for get_architecture and get_impact tools"
```

---

## Zusammenfassung F2

| Task | Datei(en) | Beschreibung | Tests |
|------|-----------|-------------|-------|
| 3 | `core/server.py` | MCP Server Skeleton mit stdio Transport, alle Tools registriert | 6 |
| 4 | `tools/code_query.py` | get_code_semantics Handler | 4 |
| 5 | `tests/` (erweitern) | get_dependencies Tests | 4 |
| 6 | `tests/` (erweitern) | get_architecture und get_impact Tests | 4 |
| **Total** | | | **~18 Tests** |

---

**Navigation:**
- Vorheriges Paket: [06a-server-foundation.md](06a-server-foundation.md) (F1 — Pydantic Models + NemesisEngine)
- Naechstes Paket: [06c-memory-index-tools.md](06c-memory-index-tools.md) (F3 — Memory + Index Tools)
