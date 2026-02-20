# 06d — Smart Context + Dispatch + Entry Point

> **Arbeitspaket F4** — Teil 4 von 4 des MCP Server Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Smart Context Tool (get_smart_context) testen, Server Integration (call_tool Dispatch) verifizieren und den CLI Entry Point fuer `nemesis serve` fertigstellen.

**Tech Stack:** Python 3.11+, MCP SDK (`mcp[cli]>=1.0`), Pydantic, Click, pytest, pytest-asyncio

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [06c-memory-index-tools.md](06c-memory-index-tools.md) (F3 — Memory + Index Tools)

**Tasks in diesem Paket:** 10, 11, 12 (von 12)

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

## Zusammenfassung F4

| Task | Datei(en) | Beschreibung | Tests |
|------|-----------|-------------|-------|
| 10 | `tests/` | Smart Context Tests (get_smart_context) | 5 |
| 11 | `tests/` (erweitern) | Server Integration: call_tool Dispatch Tests | 5 |
| 12 | `core/cli.py` (erweitern) | Entry Point, CLI-Integration, Konsistenz-Pruefung | 6 |
| **Total** | | | **~16 Tests** |

---

## Gesamtuebersicht aller Pakete

| Paket | Datei | Tasks | Tests |
|-------|-------|-------|-------|
| [F1 — Server Foundation](06a-server-foundation.md) | Models + Engine | 1, 2 | ~36 |
| [F2 — Code Intelligence](06b-code-intelligence.md) | Server + Code Tools | 3, 4, 5, 6 | ~18 |
| [F3 — Memory + Index](06c-memory-index-tools.md) | Memory + Index Tools | 7, 8, 9 | ~17 |
| **F4 — Smart + Integration** | Smart Context + CLI | 10, 11, 12 | ~16 |
| **Gesamt** | | **12 Tasks** | **~87 Tests** |

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

---

**Navigation:**
- Vorheriges Paket: [06c-memory-index-tools.md](06c-memory-index-tools.md) (F3 — Memory + Index Tools)
