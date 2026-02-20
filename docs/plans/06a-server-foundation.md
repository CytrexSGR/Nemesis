# 06a — Server Foundation (Pydantic Models + NemesisEngine)

> **Arbeitspaket F1** — Teil 1 von 4 des MCP Server Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pydantic-Modelle fuer alle 11 MCP Tool-Inputs/Outputs und das zentrale NemesisEngine Backend-Objekt erstellen.

**Tech Stack:** Python 3.11+, MCP SDK (`mcp[cli]>=1.0`), Pydantic, pytest, pytest-asyncio

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [05-indexing-pipeline](05-indexing-pipeline.md)

**Tasks in diesem Paket:** 1, 2 (von 12)

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

## Zusammenfassung F1

| Task | Datei(en) | Beschreibung | Tests |
|------|-----------|-------------|-------|
| 1 | `tools/models.py` | Pydantic-Modelle fuer alle 11 Tool-Inputs/Outputs | 30 |
| 2 | `core/engine.py` | NemesisEngine — zentrales Backend-Objekt | 6 |
| **Total** | | | **~36 Tests** |

---

**Navigation:**
- Naechstes Paket: [06b-code-intelligence.md](06b-code-intelligence.md) (F2 — Server + Code Intelligence Tools)
