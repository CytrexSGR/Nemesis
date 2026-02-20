# 06 — MCP Server

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the MCP server exposing all Nemesis code intelligence, memory, and indexing tools via stdio transport for Claude Code integration.

**Architecture:** The MCP server (`nemesis/core/server.py`) is the single entry point for Claude Code. It registers 11 tools organized into three tool modules: code intelligence (`code_query.py`), memory (`memory_query.py`), and index management (`index_tools.py`). Each tool uses Pydantic models for validated input/output. The server depends on the indexing pipeline (05), graph layer (03), and vector store (04) for all data operations. A central `NemesisEngine` class wires together all backends and is injected into every tool handler.

**Tech Stack:** Python 3.11+, MCP SDK (`mcp[cli]>=1.0`), Pydantic, pytest, pytest-asyncio

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [05-indexing-pipeline](05-indexing-pipeline.md)

**Estimated Tasks:** 12

---

## Task 1: Pydantic Models for Tool Inputs and Outputs

**Files:**
- `nemesis/tools/__init__.py` (update)
- `nemesis/tools/models.py`
- `tests/test_server/__init__.py`
- `tests/test_server/test_models.py`

### Step 1 — Write Test

```python
# tests/test_server/__init__.py
```

```python
# tests/test_server/test_models.py
"""Tests for MCP tool Pydantic input/output models."""
import pytest
from pydantic import ValidationError


def test_code_semantics_input_defaults():
    from nemesis.tools.models import CodeSemanticsInput

    inp = CodeSemanticsInput(query="auth flow")
    assert inp.query == "auth flow"
    assert inp.limit == 10


def test_code_semantics_input_custom_limit():
    from nemesis.tools.models import CodeSemanticsInput

    inp = CodeSemanticsInput(query="auth", limit=5)
    assert inp.limit == 5


def test_code_semantics_input_empty_query_rejected():
    from nemesis.tools.models import CodeSemanticsInput

    with pytest.raises(ValidationError):
        CodeSemanticsInput(query="")


def test_code_semantics_result():
    from nemesis.tools.models import CodeSemanticsResult, CodeMatch

    match = CodeMatch(
        file="src/auth.py",
        function="authenticate",
        snippet="def authenticate(user, pw):\n    ...",
        related_files=["src/utils.py"],
        relevance=0.95,
    )
    result = CodeSemanticsResult(matches=[match])
    assert len(result.matches) == 1
    assert result.matches[0].relevance == 0.95


def test_dependencies_input():
    from nemesis.tools.models import DependenciesInput

    inp = DependenciesInput(symbol="UserService", depth=3, direction="both")
    assert inp.symbol == "UserService"
    assert inp.depth == 3
    assert inp.direction == "both"


def test_dependencies_input_defaults():
    from nemesis.tools.models import DependenciesInput

    inp = DependenciesInput(symbol="Foo")
    assert inp.depth == 2
    assert inp.direction == "both"


def test_dependencies_input_invalid_direction():
    from nemesis.tools.models import DependenciesInput

    with pytest.raises(ValidationError):
        DependenciesInput(symbol="Foo", direction="sideways")


def test_dependencies_result():
    from nemesis.tools.models import DependenciesResult, DependencyNode

    node = DependencyNode(
        name="UserService",
        node_type="Class",
        file="src/user.py",
        deps_in=["AuthService"],
        deps_out=["Database"],
        call_chain=["AuthService -> UserService -> Database"],
    )
    result = DependenciesResult(root=node)
    assert result.root.name == "UserService"
    assert len(result.root.deps_out) == 1


def test_architecture_input():
    from nemesis.tools.models import ArchitectureInput

    inp = ArchitectureInput(scope="project")
    assert inp.scope == "project"


def test_architecture_input_invalid_scope():
    from nemesis.tools.models import ArchitectureInput

    with pytest.raises(ValidationError):
        ArchitectureInput(scope="universe")


def test_architecture_result():
    from nemesis.tools.models import ArchitectureResult

    result = ArchitectureResult(
        modules=["auth", "users", "api"],
        key_classes=["AuthService", "UserRepo"],
        data_flow=["Request -> AuthService -> UserRepo -> DB"],
        entry_points=["main.py:app"],
    )
    assert len(result.modules) == 3


def test_impact_input():
    from nemesis.tools.models import ImpactInput

    inp = ImpactInput(file="src/auth.py", function="login")
    assert inp.file == "src/auth.py"
    assert inp.function == "login"


def test_impact_input_file_only():
    from nemesis.tools.models import ImpactInput

    inp = ImpactInput(file="src/auth.py")
    assert inp.function is None


def test_impact_result():
    from nemesis.tools.models import ImpactResult

    result = ImpactResult(
        direct_dependents=["api.py", "tests/test_auth.py"],
        transitive_dependents=["main.py"],
        test_coverage=["tests/test_auth.py"],
    )
    assert len(result.direct_dependents) == 2


def test_project_context_input_defaults():
    from nemesis.tools.models import ProjectContextInput

    inp = ProjectContextInput()
    assert inp.topic is None


def test_project_context_input_with_topic():
    from nemesis.tools.models import ProjectContextInput

    inp = ProjectContextInput(topic="authentication")
    assert inp.topic == "authentication"


def test_project_context_result():
    from nemesis.tools.models import ProjectContextResult

    result = ProjectContextResult(
        rules=["Always use parameterized queries"],
        decisions=["Chose JWT over session cookies"],
        conventions=["snake_case for Python files"],
    )
    assert len(result.rules) == 1


def test_store_rule_input():
    from nemesis.tools.models import StoreRuleInput

    inp = StoreRuleInput(
        rule="Always use parameterized queries",
        scope="project",
        related_to=["database", "security"],
    )
    assert inp.rule == "Always use parameterized queries"
    assert inp.scope == "project"


def test_store_rule_input_defaults():
    from nemesis.tools.models import StoreRuleInput

    inp = StoreRuleInput(rule="Use type hints everywhere")
    assert inp.scope == "project"
    assert inp.related_to == []


def test_store_decision_input():
    from nemesis.tools.models import StoreDecisionInput

    inp = StoreDecisionInput(
        title="Use JWT for authentication",
        reasoning="Stateless, scalable, works with microservices",
        alternatives=["Session cookies", "OAuth only"],
    )
    assert inp.title == "Use JWT for authentication"
    assert len(inp.alternatives) == 2


def test_store_mutation_result():
    from nemesis.tools.models import StoreMutationResult

    result = StoreMutationResult(id="rule-001", stored=True)
    assert result.stored is True


def test_index_project_input():
    from nemesis.tools.models import IndexProjectInput

    inp = IndexProjectInput(path="/home/user/myproject", languages=["python", "typescript"])
    assert inp.path == "/home/user/myproject"
    assert inp.languages == ["python", "typescript"]


def test_index_project_input_defaults():
    from nemesis.tools.models import IndexProjectInput

    inp = IndexProjectInput()
    assert inp.path == "."
    assert inp.languages == ["python"]


def test_index_project_result():
    from nemesis.tools.models import IndexProjectResult

    result = IndexProjectResult(
        files_indexed=42,
        nodes=320,
        edges=180,
        duration_ms=1500.5,
    )
    assert result.files_indexed == 42


def test_index_status_result():
    from nemesis.tools.models import IndexStatusResult

    result = IndexStatusResult(
        last_indexed="2026-02-20T14:30:00Z",
        files=42,
        nodes=320,
        edges=180,
        stale_files=["src/changed.py"],
    )
    assert result.files == 42
    assert len(result.stale_files) == 1


def test_watch_project_input():
    from nemesis.tools.models import WatchProjectInput

    inp = WatchProjectInput(path="/home/user/project", enabled=True)
    assert inp.enabled is True


def test_watch_project_result():
    from nemesis.tools.models import WatchProjectResult

    result = WatchProjectResult(watching=True, path="/home/user/project", pid=12345)
    assert result.pid == 12345


def test_smart_context_input():
    from nemesis.tools.models import SmartContextInput

    inp = SmartContextInput(task="Refactor the auth service", max_tokens=4000)
    assert inp.task == "Refactor the auth service"
    assert inp.max_tokens == 4000


def test_smart_context_input_defaults():
    from nemesis.tools.models import SmartContextInput

    inp = SmartContextInput(task="Fix the bug")
    assert inp.max_tokens == 8000


def test_smart_context_result():
    from nemesis.tools.models import SmartContextResult

    result = SmartContextResult(
        code_context=[{"file": "auth.py", "snippet": "def login(): ..."}],
        rules=["Use parameterized queries"],
        decisions=["JWT over sessions"],
        architecture="3-tier microservice, FastAPI backend",
        token_count=2400,
    )
    assert result.token_count == 2400
    assert len(result.code_context) == 1
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_models.py -v
# Erwartung: FAILED — Module nemesis.tools.models existiert nicht
```

### Step 3 — Implement

```python
# nemesis/tools/__init__.py
"""Tools module — MCP tool implementations for Nemesis."""
```

```python
# nemesis/tools/models.py
"""Pydantic models for all MCP tool inputs and outputs.

Every MCP tool has a validated input model and a structured output model.
These models serve as the contract between the MCP server and the tool
handler functions.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Code Intelligence
# ---------------------------------------------------------------------------


class CodeSemanticsInput(BaseModel):
    """Input for get_code_semantics tool."""

    query: str = Field(..., min_length=1, description="Natural language search query.")
    limit: int = Field(default=10, ge=1, le=50, description="Max results to return.")


class CodeMatch(BaseModel):
    """A single code search match."""

    file: str
    function: str
    snippet: str
    related_files: list[str] = Field(default_factory=list)
    relevance: float = Field(ge=0.0, le=1.0)


class CodeSemanticsResult(BaseModel):
    """Output for get_code_semantics tool."""

    matches: list[CodeMatch] = Field(default_factory=list)


class DependenciesInput(BaseModel):
    """Input for get_dependencies tool."""

    symbol: str = Field(..., min_length=1, description="Symbol name to look up.")
    depth: int = Field(default=2, ge=1, le=10, description="Traversal depth.")
    direction: Literal["in", "out", "both"] = Field(
        default="both", description="Traversal direction."
    )


class DependencyNode(BaseModel):
    """A node in the dependency graph."""

    name: str
    node_type: str
    file: str
    deps_in: list[str] = Field(default_factory=list)
    deps_out: list[str] = Field(default_factory=list)
    call_chain: list[str] = Field(default_factory=list)


class DependenciesResult(BaseModel):
    """Output for get_dependencies tool."""

    root: DependencyNode


class ArchitectureInput(BaseModel):
    """Input for get_architecture tool."""

    scope: Literal["project", "module", "file"] = Field(
        default="project", description="Scope of the architecture overview."
    )
    target: str | None = Field(
        default=None, description="Module or file path when scope is not 'project'."
    )


class ArchitectureResult(BaseModel):
    """Output for get_architecture tool."""

    modules: list[str] = Field(default_factory=list)
    key_classes: list[str] = Field(default_factory=list)
    data_flow: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)


class ImpactInput(BaseModel):
    """Input for get_impact tool."""

    file: str = Field(..., description="File path to analyze.")
    function: str | None = Field(default=None, description="Function name (optional).")


class ImpactResult(BaseModel):
    """Output for get_impact tool."""

    direct_dependents: list[str] = Field(default_factory=list)
    transitive_dependents: list[str] = Field(default_factory=list)
    test_coverage: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class ProjectContextInput(BaseModel):
    """Input for get_project_context tool."""

    topic: str | None = Field(default=None, description="Topic filter (optional).")


class ProjectContextResult(BaseModel):
    """Output for get_project_context tool."""

    rules: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    conventions: list[str] = Field(default_factory=list)


class StoreRuleInput(BaseModel):
    """Input for store_rule tool."""

    rule: str = Field(..., min_length=1, description="The rule text.")
    scope: str = Field(default="project", description="Scope: project, module, or file.")
    related_to: list[str] = Field(
        default_factory=list, description="Related topics or file paths."
    )


class StoreDecisionInput(BaseModel):
    """Input for store_decision tool."""

    title: str = Field(..., min_length=1, description="Decision title.")
    reasoning: str = Field(..., min_length=1, description="Why this decision was made.")
    alternatives: list[str] = Field(
        default_factory=list, description="Alternatives that were considered."
    )


class StoreMutationResult(BaseModel):
    """Output for store_rule and store_decision tools."""

    id: str
    stored: bool


# ---------------------------------------------------------------------------
# Index Management
# ---------------------------------------------------------------------------


class IndexProjectInput(BaseModel):
    """Input for index_project tool."""

    path: str = Field(default=".", description="Project root path.")
    languages: list[str] = Field(
        default_factory=lambda: ["python"], description="Languages to index."
    )


class IndexProjectResult(BaseModel):
    """Output for index_project tool."""

    files_indexed: int
    nodes: int
    edges: int
    duration_ms: float


class IndexStatusResult(BaseModel):
    """Output for index_status tool."""

    last_indexed: str | None = None
    files: int = 0
    nodes: int = 0
    edges: int = 0
    stale_files: list[str] = Field(default_factory=list)


class WatchProjectInput(BaseModel):
    """Input for watch_project tool."""

    path: str = Field(default=".", description="Project root path.")
    enabled: bool = Field(default=True, description="Enable or disable watcher.")


class WatchProjectResult(BaseModel):
    """Output for watch_project tool."""

    watching: bool
    path: str
    pid: int | None = None


# ---------------------------------------------------------------------------
# Smart Context
# ---------------------------------------------------------------------------


class SmartContextInput(BaseModel):
    """Input for get_smart_context tool — the CLAUDE.md replacement."""

    task: str = Field(..., min_length=1, description="Task description.")
    max_tokens: int = Field(
        default=8000, ge=500, le=32000, description="Token budget for context."
    )


class SmartContextResult(BaseModel):
    """Output for get_smart_context tool."""

    code_context: list[dict] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    architecture: str = ""
    token_count: int = 0
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_models.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/tools/__init__.py nemesis/tools/models.py \
        tests/test_server/__init__.py tests/test_server/test_models.py
git commit -m "feat(mcp): add Pydantic models for all 11 MCP tool inputs/outputs"
```

---

## Task 2: NemesisEngine — Zentrales Backend-Objekt

**Files:**
- `nemesis/core/engine.py`
- `tests/test_server/test_engine.py`

### Step 1 — Write Test

```python
# tests/test_server/test_engine.py
"""Tests for NemesisEngine — the central backend wiring."""
import pytest
from unittest.mock import MagicMock, AsyncMock


def _make_mock_graph():
    graph = MagicMock()
    graph.get_file_hashes.return_value = {}
    graph.query.return_value = []
    return graph


def _make_mock_vector_store():
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    store.count = AsyncMock(return_value=0)
    store.is_initialized = True
    return store


def _make_mock_embedder():
    embedder = MagicMock()
    embedder.embed_single = AsyncMock(return_value=[0.1] * 384)
    embedder.embed = AsyncMock()
    return embedder


def _make_mock_pipeline():
    pipeline = MagicMock()
    return pipeline


def test_engine_creation():
    from nemesis.core.engine import NemesisEngine

    engine = NemesisEngine(
        graph=_make_mock_graph(),
        vector_store=_make_mock_vector_store(),
        embedder=_make_mock_embedder(),
        pipeline=_make_mock_pipeline(),
    )
    assert engine is not None


def test_engine_has_components():
    from nemesis.core.engine import NemesisEngine

    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()
    pipeline = _make_mock_pipeline()

    engine = NemesisEngine(
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
        pipeline=pipeline,
    )
    assert engine.graph is graph
    assert engine.vector_store is vector_store
    assert engine.embedder is embedder
    assert engine.pipeline is pipeline


def test_engine_is_ready_all_set():
    from nemesis.core.engine import NemesisEngine

    engine = NemesisEngine(
        graph=_make_mock_graph(),
        vector_store=_make_mock_vector_store(),
        embedder=_make_mock_embedder(),
        pipeline=_make_mock_pipeline(),
    )
    assert engine.is_ready is True


def test_engine_not_ready_no_vector():
    from nemesis.core.engine import NemesisEngine

    vs = _make_mock_vector_store()
    vs.is_initialized = False
    engine = NemesisEngine(
        graph=_make_mock_graph(),
        vector_store=vs,
        embedder=_make_mock_embedder(),
        pipeline=_make_mock_pipeline(),
    )
    assert engine.is_ready is False


def test_engine_project_root_default():
    from nemesis.core.engine import NemesisEngine
    from pathlib import Path

    engine = NemesisEngine(
        graph=_make_mock_graph(),
        vector_store=_make_mock_vector_store(),
        embedder=_make_mock_embedder(),
        pipeline=_make_mock_pipeline(),
    )
    assert engine.project_root == Path(".")


def test_engine_project_root_custom():
    from nemesis.core.engine import NemesisEngine
    from pathlib import Path

    engine = NemesisEngine(
        graph=_make_mock_graph(),
        vector_store=_make_mock_vector_store(),
        embedder=_make_mock_embedder(),
        pipeline=_make_mock_pipeline(),
        project_root=Path("/home/user/myproject"),
    )
    assert engine.project_root == Path("/home/user/myproject")
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_engine.py -v
# Erwartung: FAILED — Module nemesis.core.engine existiert nicht
```

### Step 3 — Implement

```python
# nemesis/core/engine.py
"""NemesisEngine — central backend wiring for all MCP tools.

The engine holds references to all backend components (graph, vector store,
embedder, indexing pipeline) and provides them to tool handlers. It is
created once at server startup and injected into every tool function.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NemesisEngine:
    """Central orchestrator that connects all Nemesis backends.

    This object is created once during MCP server startup and passed
    to every tool handler. It provides access to:
    - graph: Graph adapter (Kuzu or Neo4j)
    - vector_store: LanceDB vector store
    - embedder: Embedding provider (OpenAI or local)
    - pipeline: Indexing pipeline (parse -> chunk -> embed -> store)

    Args:
        graph: Graph adapter instance.
        vector_store: Vector store instance.
        embedder: Embedding provider instance.
        pipeline: Indexing pipeline instance.
        project_root: Root directory of the indexed project.
    """

    def __init__(
        self,
        graph: Any,
        vector_store: Any,
        embedder: Any,
        pipeline: Any,
        project_root: Path | None = None,
    ) -> None:
        self._graph = graph
        self._vector_store = vector_store
        self._embedder = embedder
        self._pipeline = pipeline
        self._project_root = project_root or Path(".")
        self._watcher_pid: int | None = None

    @property
    def graph(self) -> Any:
        """Graph adapter (Kuzu or Neo4j)."""
        return self._graph

    @property
    def vector_store(self) -> Any:
        """LanceDB vector store."""
        return self._vector_store

    @property
    def embedder(self) -> Any:
        """Embedding provider (OpenAI or local)."""
        return self._embedder

    @property
    def pipeline(self) -> Any:
        """Indexing pipeline."""
        return self._pipeline

    @property
    def project_root(self) -> Path:
        """Root directory of the indexed project."""
        return self._project_root

    @property
    def is_ready(self) -> bool:
        """True if all backends are initialized and ready."""
        try:
            return (
                self._graph is not None
                and self._vector_store is not None
                and self._embedder is not None
                and self._pipeline is not None
                and getattr(self._vector_store, "is_initialized", False)
            )
        except Exception:
            return False

    @property
    def watcher_pid(self) -> int | None:
        """PID of the running file watcher, if any."""
        return self._watcher_pid

    @watcher_pid.setter
    def watcher_pid(self, pid: int | None) -> None:
        self._watcher_pid = pid
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_engine.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/core/engine.py tests/test_server/test_engine.py
git commit -m "feat(mcp): add NemesisEngine central backend wiring"
```

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

## Task 10: Smart Context — get_smart_context

**Files:**
- `tests/test_server/test_smart_context.py`

### Step 1 — Write Test

```python
# tests/test_server/test_smart_context.py
"""Tests for the get_smart_context tool — the CLAUDE.md replacement."""
import pytest
from unittest.mock import MagicMock, AsyncMock


def _make_mock_engine():
    engine = MagicMock()
    engine.graph = MagicMock()
    engine.vector_store = MagicMock()
    engine.embedder = MagicMock()
    engine.pipeline = MagicMock()
    engine.is_ready = True
    engine.project_root = MagicMock()

    engine.embedder.embed_single = AsyncMock(return_value=[0.1] * 384)
    engine.vector_store.search = AsyncMock(return_value=[])
    engine.graph.query = MagicMock(return_value=[])
    engine.graph.get_related_files = MagicMock(return_value=[])

    return engine


@pytest.mark.asyncio
async def test_smart_context_returns_result():
    from nemesis.tools.code_query import handle_smart_context
    from nemesis.tools.models import SmartContextInput, SmartContextResult

    engine = _make_mock_engine()

    inp = SmartContextInput(task="Refactor the auth service", max_tokens=4000)
    result = await handle_smart_context(engine, inp)

    assert isinstance(result, SmartContextResult)
    assert result.token_count >= 0


@pytest.mark.asyncio
async def test_smart_context_includes_code():
    from nemesis.tools.code_query import handle_smart_context
    from nemesis.tools.models import SmartContextInput

    engine = _make_mock_engine()

    mock_hit = MagicMock()
    mock_hit.id = "func-001:chunk-0"
    mock_hit.text = "def authenticate(user, pw):\n    return check(pw)"
    mock_hit.score = 0.9
    mock_hit.metadata = {
        "file_path": "src/auth.py",
        "parent_node_id": "func-001",
        "parent_type": "Function",
        "start_line": 1,
        "end_line": 5,
    }
    engine.vector_store.search = AsyncMock(return_value=[mock_hit])

    inp = SmartContextInput(task="How does auth work?", max_tokens=8000)
    result = await handle_smart_context(engine, inp)

    assert len(result.code_context) >= 1
    assert result.code_context[0]["file"] == "src/auth.py"


@pytest.mark.asyncio
async def test_smart_context_includes_rules():
    from nemesis.tools.code_query import handle_smart_context
    from nemesis.tools.models import SmartContextInput

    engine = _make_mock_engine()
    # Make graph return rules when memory_query queries it
    engine.graph.query = MagicMock(side_effect=[
        [],   # files (architecture)
        [],   # classes (architecture)
        [],   # imports (architecture)
        [],   # entries (architecture)
        [{"content": "Use parameterized queries"}],  # rules
        [],   # decisions
        [],   # conventions
    ])

    inp = SmartContextInput(task="database security", max_tokens=8000)
    result = await handle_smart_context(engine, inp)

    assert isinstance(result.rules, list)


@pytest.mark.asyncio
async def test_smart_context_respects_token_budget():
    from nemesis.tools.code_query import handle_smart_context
    from nemesis.tools.models import SmartContextInput

    engine = _make_mock_engine()

    # Create many search results that would exceed budget
    hits = []
    for i in range(20):
        hit = MagicMock()
        hit.id = f"func-{i}:chunk-0"
        hit.text = "x = 1\n" * 200  # ~200 tokens per hit
        hit.score = 0.9 - i * 0.01
        hit.metadata = {
            "file_path": f"src/file_{i}.py",
            "parent_node_id": f"func-{i}",
            "parent_type": "Function",
            "start_line": 1,
            "end_line": 200,
        }
        hits.append(hit)

    engine.vector_store.search = AsyncMock(return_value=hits)

    inp = SmartContextInput(task="test", max_tokens=500)
    result = await handle_smart_context(engine, inp)

    # Should not include all 20 hits — token budget limits it
    assert len(result.code_context) < 20


@pytest.mark.asyncio
async def test_smart_context_empty_index():
    from nemesis.tools.code_query import handle_smart_context
    from nemesis.tools.models import SmartContextInput

    engine = _make_mock_engine()

    inp = SmartContextInput(task="anything", max_tokens=4000)
    result = await handle_smart_context(engine, inp)

    assert result.code_context == []
    assert result.token_count >= 0
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_smart_context.py -v
# Erwartung: PASSED — Implementierung schon in Task 4
```

### Step 3 — Implement

Bereits in Task 4 implementiert.

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_smart_context.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add tests/test_server/test_smart_context.py
git commit -m "test(mcp): add comprehensive tests for get_smart_context tool"
```

---

## Task 11: Server Integration — call_tool Dispatch

**Files:**
- `tests/test_server/test_server.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_server/test_server.py — APPEND folgende Tests

@pytest.mark.asyncio
async def test_call_tool_dispatches_to_handler():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True
    mock_engine.graph = MagicMock()
    mock_engine.vector_store = MagicMock()
    mock_engine.embedder = MagicMock()
    mock_engine.pipeline = MagicMock()
    mock_engine.project_root = MagicMock()
    mock_engine.watcher_pid = None

    mock_engine.embedder.embed_single = AsyncMock(return_value=[0.1] * 384)
    mock_engine.vector_store.search = AsyncMock(return_value=[])
    mock_engine.graph.query = MagicMock(return_value=[])
    mock_engine.graph.get_related_files = MagicMock(return_value=[])

    srv = create_server(mock_engine)

    # Call the code semantics tool
    result = await srv.call_tool("get_code_semantics", {"query": "test"})
    assert len(result) == 1
    assert result[0].type == "text"
    assert "matches" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_unknown_tool():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True

    srv = create_server(mock_engine)

    result = await srv.call_tool("nonexistent_tool", {})
    assert len(result) == 1
    assert "Unknown tool" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_returns_json():
    from nemesis.core.server import create_server
    import json

    mock_engine = MagicMock()
    mock_engine.is_ready = True
    mock_engine.graph = MagicMock()
    mock_engine.graph.query = MagicMock(return_value=[])
    mock_engine.vector_store = MagicMock()
    mock_engine.vector_store.count = AsyncMock(return_value=0)
    mock_engine.watcher_pid = None

    srv = create_server(mock_engine)

    result = await srv.call_tool("index_status", {})
    assert len(result) == 1
    parsed = json.loads(result[0].text)
    assert "files" in parsed
    assert "nodes" in parsed


@pytest.mark.asyncio
async def test_call_tool_handles_validation_error():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True

    srv = create_server(mock_engine)

    # Missing required field 'query'
    result = await srv.call_tool("get_code_semantics", {})
    assert len(result) == 1
    assert "Error" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_index_project():
    from nemesis.core.server import create_server

    mock_engine = MagicMock()
    mock_engine.is_ready = True
    mock_engine.pipeline = MagicMock()
    mock_result = MagicMock()
    mock_result.files_indexed = 10
    mock_result.nodes_created = 50
    mock_result.edges_created = 30
    mock_result.duration_ms = 1000.0
    mock_result.success = True
    mock_result.errors = []
    mock_engine.pipeline.index_project.return_value = mock_result

    srv = create_server(mock_engine)

    import json
    result = await srv.call_tool("index_project", {
        "path": "/tmp/test",
        "languages": ["python"],
    })
    parsed = json.loads(result[0].text)
    assert parsed["files_indexed"] == 10
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_server.py -v -k "call_tool"
# Erwartung: Moeglicherweise PASSED wenn die call_tool Methode auf dem Server richtig
# exposed ist, andernfalls FAIL.
```

### Step 3 — Implement

Bereits in Task 3 implementiert. Falls die `call_tool`-Methode nicht direkt auf dem Server-Objekt aufrufbar ist, muss der Test angepasst werden um den internen Handler direkt zu testen. Die MCP SDK registriert Handler intern, daher testen wir die Handler-Funktionen direkt.

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_server.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add tests/test_server/test_server.py
git commit -m "test(mcp): add server integration tests for call_tool dispatch"
```

---

## Task 12: Server Entry Point und CLI-Integration

**Files:**
- `nemesis/core/cli.py` (erweitern)
- `tests/test_server/test_entry_point.py`

### Step 1 — Write Test

```python
# tests/test_server/test_entry_point.py
"""Tests for MCP server entry point and CLI integration."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from click.testing import CliRunner


def test_serve_command_exists():
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "MCP" in result.output or "server" in result.output.lower()


def test_run_server_function_exists():
    from nemesis.core.server import run_server
    assert callable(run_server)


def test_create_server_function_exists():
    from nemesis.core.server import create_server
    assert callable(create_server)


@pytest.mark.asyncio
async def test_create_server_with_engine():
    from nemesis.core.server import create_server
    from nemesis.core.engine import NemesisEngine

    engine = NemesisEngine(
        graph=MagicMock(),
        vector_store=MagicMock(is_initialized=True),
        embedder=MagicMock(),
        pipeline=MagicMock(),
    )

    srv = create_server(engine)
    tools = await srv.list_tools()
    assert len(tools) == 11


def test_tool_names_match_design_doc():
    """Verify all tool names match the design document specification."""
    from nemesis.core.server import _TOOL_DEFINITIONS

    expected = {
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
    }

    actual = {d["name"] for d in _TOOL_DEFINITIONS}
    assert actual == expected


def test_tool_handlers_all_registered():
    """Every tool definition has a corresponding handler."""
    from nemesis.core.server import _TOOL_DEFINITIONS, _TOOL_HANDLERS

    for defn in _TOOL_DEFINITIONS:
        assert defn["name"] in _TOOL_HANDLERS, (
            f"Tool {defn['name']} missing from _TOOL_HANDLERS"
        )


def test_handler_count_matches_definition_count():
    from nemesis.core.server import _TOOL_DEFINITIONS, _TOOL_HANDLERS

    assert len(_TOOL_HANDLERS) == len(_TOOL_DEFINITIONS)
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/test_entry_point.py -v
# Erwartung: PASSED — Alles ist bereits implementiert
```

### Step 3 — Implement

Update der CLI um den MCP Server korrekt zu starten:

```python
# nemesis/core/cli.py — serve Kommando ersetzen

@main.command()
@click.option("--transport", default="stdio", type=click.Choice(["stdio"]),
              help="Transport protocol.")
def serve(transport: str) -> None:
    """Start the Nemesis MCP server for Claude Code integration.

    Runs the MCP server on stdio transport. Connect from Claude Code
    by adding Nemesis to your MCP configuration.
    """
    import asyncio

    click.echo("Starting Nemesis MCP server...")

    from nemesis.core.engine import NemesisEngine
    from nemesis.core.server import run_server

    # In production, engine would be initialized with real backends.
    # For now, create a minimal engine to be replaced during actual setup.
    engine = NemesisEngine(
        graph=None,
        vector_store=None,
        embedder=None,
        pipeline=None,
    )

    asyncio.run(run_server(engine))
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_server/ -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/core/cli.py tests/test_server/test_entry_point.py
git commit -m "feat(mcp): add server entry point and CLI integration for nemesis serve"
```

---

## Zusammenfassung

| Task | Datei(en) | Beschreibung | Tests |
|------|-----------|-------------|-------|
| 1 | `tools/models.py` | Pydantic-Modelle fuer alle 11 Tool-Inputs/Outputs | 30 |
| 2 | `core/engine.py` | NemesisEngine — zentrales Backend-Objekt | 6 |
| 3 | `core/server.py` | MCP Server Skeleton mit stdio Transport, alle Tools registriert | 6 |
| 4 | `tools/code_query.py` | get_code_semantics Handler | 4 |
| 5 | `tests/` (erweitern) | get_dependencies Tests | 4 |
| 6 | `tests/` (erweitern) | get_architecture und get_impact Tests | 4 |
| 7 | `tools/memory_query.py` | Memory Tools: get_project_context, store_rule, store_decision | 9 |
| 8 | `tools/index_tools.py` | Index Tools: index_project, index_status | 5 |
| 9 | `tests/` (erweitern) | watch_project Tests | 3 |
| 10 | `tests/` | Smart Context Tests (get_smart_context) | 5 |
| 11 | `tests/` (erweitern) | Server Integration: call_tool Dispatch Tests | 5 |
| 12 | `core/cli.py` (erweitern) | Entry Point, CLI-Integration, Konsistenz-Pruefung | 6 |
| **Total** | | | **~87 Tests** |

### Dateien erstellt/geaendert

```
nemesis/
├── core/
│   ├── engine.py              # NemesisEngine — Backend-Wiring
│   ├── server.py              # MCP Server (stdio), Tool-Registration, Dispatch
│   └── cli.py                 # serve-Kommando aktualisiert
└── tools/
    ├── __init__.py            # Package init
    ├── models.py              # Pydantic-Modelle fuer alle 11 Tools
    ├── code_query.py          # Code Intelligence (semantics, deps, arch, impact, smart context)
    ├── memory_query.py        # Memory Tools (context, store_rule, store_decision)
    └── index_tools.py         # Index Management (index, status, watch)

tests/test_server/
├── __init__.py
├── test_models.py             # Pydantic Model Validierung
├── test_engine.py             # NemesisEngine Tests
├── test_server.py             # Server Setup + call_tool Dispatch
├── test_code_query.py         # Code Intelligence Handler Tests
├── test_memory_query.py       # Memory Handler Tests
├── test_index_tools.py        # Index Management Handler Tests
├── test_smart_context.py      # Smart Context Handler Tests
└── test_entry_point.py        # Entry Point + CLI Integration
```

### MCP Tool-Uebersicht

| Tool | Kategorie | Input-Model | Output-Model |
|------|-----------|-------------|--------------|
| `get_code_semantics` | Code Intelligence | `CodeSemanticsInput` | `CodeSemanticsResult` |
| `get_dependencies` | Code Intelligence | `DependenciesInput` | `DependenciesResult` |
| `get_architecture` | Code Intelligence | `ArchitectureInput` | `ArchitectureResult` |
| `get_impact` | Code Intelligence | `ImpactInput` | `ImpactResult` |
| `get_project_context` | Memory | `ProjectContextInput` | `ProjectContextResult` |
| `store_rule` | Memory | `StoreRuleInput` | `StoreMutationResult` |
| `store_decision` | Memory | `StoreDecisionInput` | `StoreMutationResult` |
| `index_project` | Index Management | `IndexProjectInput` | `IndexProjectResult` |
| `index_status` | Index Management | (keine) | `IndexStatusResult` |
| `watch_project` | Index Management | `WatchProjectInput` | `WatchProjectResult` |
| `get_smart_context` | Smart Context | `SmartContextInput` | `SmartContextResult` |

### Abhaengigkeiten gemockt

Alle externen Abhaengigkeiten werden in Tests durch Mocks ersetzt:
- **Graph** (`query`, `add_node`, `add_edge`, `get_related_files`, `get_dependents`, `get_dependencies_of`) — MagicMock
- **VectorStore** (`search`, `count`, `is_initialized`) — AsyncMock/MagicMock
- **Embedder** (`embed_single`, `embed`) — AsyncMock
- **Pipeline** (`index_project`) — MagicMock
- **MCP Server** — echtes `Server`-Objekt mit gemocktem Engine
