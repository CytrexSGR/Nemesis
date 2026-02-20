# Multi-Project Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Nemesis support multiple projects in a single engine instance with namespaced IDs and a central `~/.nemesis/` data directory.

**Architecture:** Single KuzuDB + single LanceDB under `~/.nemesis/`. Node IDs prefixed with `project_id::`, vector chunks tagged with `project_id` field. A `ProjectRegistry` manages known projects in `~/.nemesis/registry.json`. All tools get an optional `project` parameter; cross-project search is default when no project is specified.

**Tech Stack:** Python 3.12, Pydantic, KuzuDB, LanceDB, Click CLI, MCP SDK

---

### Task 1: ProjectRegistry

**Files:**
- Create: `nemesis/core/registry.py`
- Test: `tests/test_core/test_registry.py`

**Step 1: Write the failing tests**

```python
# tests/test_core/test_registry.py
"""Tests for ProjectRegistry."""

import json
from pathlib import Path

import pytest

from nemesis.core.registry import ProjectInfo, ProjectRegistry


@pytest.fixture
def registry(tmp_path):
    return ProjectRegistry(tmp_path / "registry.json")


class TestProjectRegistry:
    def test_register_creates_entry(self, registry):
        info = registry.register(
            path=Path("/home/user/projects/eve"),
            languages=["python", "typescript"],
        )
        assert info.name == "eve"
        assert info.path == Path("/home/user/projects/eve")
        assert info.languages == ["python", "typescript"]

    def test_register_custom_name(self, registry):
        info = registry.register(
            path=Path("/home/user/projects/Eve-Online-Copilot"),
            name="eve",
            languages=["python"],
        )
        assert info.name == "eve"

    def test_register_default_name_from_dirname(self, registry):
        info = registry.register(
            path=Path("/home/user/projects/my-cool-project"),
            languages=["python"],
        )
        assert info.name == "my-cool-project"

    def test_register_persists_to_file(self, registry, tmp_path):
        registry.register(path=Path("/tmp/proj"), languages=["python"])
        data = json.loads((tmp_path / "registry.json").read_text())
        assert "proj" in data["projects"]

    def test_unregister_removes_entry(self, registry):
        registry.register(path=Path("/tmp/proj"), languages=["python"])
        registry.unregister("proj")
        assert registry.get("proj") is None

    def test_unregister_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.unregister("nope")

    def test_list_projects(self, registry):
        registry.register(path=Path("/tmp/a"), languages=["python"])
        registry.register(path=Path("/tmp/b"), languages=["rust"])
        projects = registry.list_projects()
        assert len(projects) == 2
        assert "a" in projects
        assert "b" in projects

    def test_resolve_finds_project_by_subpath(self, registry):
        registry.register(path=Path("/home/user/projects/eve"), languages=["python"])
        result = registry.resolve(Path("/home/user/projects/eve/services/main.py"))
        assert result == "eve"

    def test_resolve_returns_none_for_unknown(self, registry):
        assert registry.resolve(Path("/unknown/path/file.py")) is None

    def test_get_returns_project_info(self, registry):
        registry.register(path=Path("/tmp/proj"), languages=["python"])
        info = registry.get("proj")
        assert info is not None
        assert info.name == "proj"

    def test_get_returns_none_for_unknown(self, registry):
        assert registry.get("nope") is None

    def test_update_indexed_at(self, registry):
        registry.register(path=Path("/tmp/proj"), languages=["python"])
        registry.update_stats("proj", files=42)
        info = registry.get("proj")
        assert info.files == 42
        assert info.indexed_at is not None

    def test_registry_loads_from_existing_file(self, tmp_path):
        data = {
            "projects": {
                "old": {
                    "path": "/tmp/old",
                    "languages": ["python"],
                    "indexed_at": None,
                    "files": 0,
                }
            }
        }
        (tmp_path / "registry.json").write_text(json.dumps(data))
        reg = ProjectRegistry(tmp_path / "registry.json")
        assert reg.get("old") is not None
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_core/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nemesis.core.registry'`

**Step 3: Write the implementation**

```python
# nemesis/core/registry.py
"""Project Registry — manages registered projects in ~/.nemesis/registry.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ProjectInfo:
    """Info about a registered project."""

    name: str
    path: Path
    languages: list[str] = field(default_factory=list)
    indexed_at: str | None = None
    files: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict) -> ProjectInfo:
        return cls(
            name=name,
            path=Path(data["path"]),
            languages=data.get("languages", []),
            indexed_at=data.get("indexed_at"),
            files=data.get("files", 0),
        )


class ProjectRegistry:
    """Manages registered projects persisted to a JSON file."""

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path
        self._projects: dict[str, ProjectInfo] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for name, proj_data in data.get("projects", {}).items():
                self._projects[name] = ProjectInfo.from_dict(name, proj_data)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"projects": {name: info.to_dict() for name, info in self._projects.items()}}
        self._path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        path: Path,
        languages: list[str],
        name: str | None = None,
    ) -> ProjectInfo:
        project_name = name or path.name
        info = ProjectInfo(name=project_name, path=path.resolve(), languages=languages)
        self._projects[project_name] = info
        self._save()
        return info

    def unregister(self, name: str) -> None:
        if name not in self._projects:
            raise KeyError(f"Project '{name}' not found in registry")
        del self._projects[name]
        self._save()

    def list_projects(self) -> dict[str, ProjectInfo]:
        return dict(self._projects)

    def get(self, name: str) -> ProjectInfo | None:
        return self._projects.get(name)

    def resolve(self, file_path: Path) -> str | None:
        resolved = file_path.resolve()
        for name, info in self._projects.items():
            try:
                resolved.relative_to(info.path)
                return name
            except ValueError:
                continue
        return None

    def update_stats(self, name: str, files: int) -> None:
        info = self._projects.get(name)
        if info is None:
            raise KeyError(f"Project '{name}' not found")
        info.files = files
        info.indexed_at = datetime.now(timezone.utc).isoformat()
        self._save()
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_core/test_registry.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add nemesis/core/registry.py tests/test_core/test_registry.py
git commit -m "feat: add ProjectRegistry for multi-project support"
```

---

### Task 2: NemesisConfig — central data_dir

**Files:**
- Modify: `nemesis/core/config.py`
- Modify: `tests/test_core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_core/test_config.py`:

```python
def test_data_dir_defaults_to_home_nemesis():
    config = NemesisConfig()
    assert config.data_dir == Path.home() / ".nemesis"

def test_graph_dir(self):
    config = NemesisConfig()
    assert config.graph_dir == config.data_dir / "graph"

def test_vector_dir(self):
    config = NemesisConfig()
    assert config.vector_dir == config.data_dir / "vectors"
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_core/test_config.py -v -k "data_dir or graph_dir or vector_dir"`
Expected: FAIL

**Step 3: Modify config.py**

Replace the entire `NemesisConfig` class. Key changes:
- Remove `project_root`, `graph_path`, `vector_path`
- Add `data_dir: Path = Path.home() / ".nemesis"`
- Add properties `graph_dir` and `vector_dir`

```python
class NemesisConfig(BaseSettings):
    model_config = {
        "env_prefix": "NEMESIS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # Central data directory
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".nemesis")

    # Defaults for new projects
    languages: list[str] = Field(default_factory=lambda: ["python"])
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules", "venv", ".venv", "__pycache__", ".git",
            "target", "dist", "build",
        ]
    )

    # Graph DB
    graph_backend: Literal["kuzu", "neo4j"] = "kuzu"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Vector DB / Embeddings
    vector_provider: Literal["openai", "local"] = "openai"
    vector_model: str = "text-embedding-3-small"
    openai_api_key: str = ""

    # Memory
    memory_auto_load_rules: bool = True
    memory_auto_learn: bool = True

    # Watcher
    watcher_enabled: bool = True
    watcher_debounce_ms: int = 500

    @property
    def graph_dir(self) -> Path:
        return self.data_dir / "graph"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"
```

**Step 4: Run tests — fix any broken tests that relied on project_root**

Run: `.venv/bin/python -m pytest tests/test_core/test_config.py -v`
Expected: PASS (may need to update old tests that used `project_root`)

**Step 5: Commit**

```bash
git add nemesis/core/config.py tests/test_core/test_config.py
git commit -m "refactor: centralize config to ~/.nemesis data_dir"
```

---

### Task 3: VectorStore — add project_id field

**Files:**
- Modify: `nemesis/vector/store.py`
- Modify: `tests/test_vector/test_store.py`

**Step 1: Write the failing test**

Add to `tests/test_vector/test_store.py`:

```python
@pytest.mark.asyncio
async def test_add_with_project_id(store):
    await store.add(
        ids=["id1"],
        texts=["hello world"],
        embeddings=[[0.1] * 16],
        metadata=[{"file": "a.py"}],
        project_id="eve",
    )
    results = await store.search([0.1] * 16, limit=10, project_id="eve")
    assert len(results) == 1
    assert results[0].project_id == "eve"

@pytest.mark.asyncio
async def test_search_filters_by_project(store):
    await store.add(["id1"], ["hello"], [[0.1] * 16], [{}], project_id="eve")
    await store.add(["id2"], ["world"], [[0.1] * 16], [{}], project_id="nemesis")
    eve_results = await store.search([0.1] * 16, limit=10, project_id="eve")
    assert all(r.project_id == "eve" for r in eve_results)

@pytest.mark.asyncio
async def test_search_all_projects(store):
    await store.add(["id1"], ["hello"], [[0.1] * 16], [{}], project_id="eve")
    await store.add(["id2"], ["world"], [[0.1] * 16], [{}], project_id="nemesis")
    all_results = await store.search([0.1] * 16, limit=10)
    assert len(all_results) == 2

@pytest.mark.asyncio
async def test_delete_by_project(store):
    await store.add(["id1"], ["hello"], [[0.1] * 16], [{}], project_id="eve")
    await store.add(["id2"], ["world"], [[0.1] * 16], [{}], project_id="nemesis")
    await store.delete_by_project("eve")
    assert await store.count() == 1
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_vector/test_store.py -v -k "project"`
Expected: FAIL

**Step 3: Modify store.py**

Key changes:
- Add `project_id` field to schema (pa.field)
- `add()` gets `project_id: str = ""` parameter — stored as column
- `search()` gets `project_id: str | None = None` — filters with WHERE clause when set
- `SearchResult` gets `project_id: str = ""` field
- Add `delete_by_project(project_id: str)` method
- `SyncVectorStoreWrapper` in engine.py gets matching methods

Schema change in `initialize()`:
```python
schema = pa.schema([
    pa.field("id", pa.utf8()),
    pa.field("text", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), list_size=dimensions)),
    pa.field("metadata", pa.utf8()),
    pa.field("project_id", pa.utf8()),
])
```

`add()` — add project_id to rows:
```python
async def add(self, ids, texts, embeddings, metadata, project_id: str = "") -> None:
    rows = [
        {
            "id": id_, "text": text, "vector": embedding,
            "metadata": json.dumps(metadata_item),
            "project_id": project_id,
        }
        for id_, text, embedding, metadata_item in zip(ids, texts, embeddings, metadata, strict=True)
    ]
```

`search()` — filter by project_id:
```python
async def search(self, query_embedding, limit=10, filter=None, project_id: str | None = None):
    def _search():
        query = self._table.search(query_embedding).limit(limit)
        conditions = []
        if project_id is not None:
            conditions.append(f"project_id = '{project_id}'")
        if filter:
            # existing metadata filter logic
            ...
        if conditions:
            query = query.where(" AND ".join(conditions))
        return query.to_list()
```

`delete_by_project()`:
```python
async def delete_by_project(self, project_id: str) -> None:
    self._require_initialized()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: self._table.delete(f"project_id = '{project_id}'")
    )
```

**Step 4: Run all vector store tests**

Run: `.venv/bin/python -m pytest tests/test_vector/test_store.py -v`
Expected: all PASS

**Step 5: Update SyncVectorStoreWrapper in engine.py**

Add `project_id` params to `add()`, `search()`, and new `delete_by_project()` method.

**Step 6: Commit**

```bash
git add nemesis/vector/store.py nemesis/core/engine.py tests/test_vector/test_store.py
git commit -m "feat: add project_id field to VectorStore"
```

---

### Task 4: Pipeline — project_id and relative paths

**Files:**
- Modify: `nemesis/indexer/pipeline.py`
- Modify: `nemesis/indexer/chunker.py`
- Modify: `tests/test_indexer/test_pipeline.py`

**Step 1: Write the failing test**

```python
def test_index_file_uses_project_id(mock_pipeline):
    """Node IDs and chunk IDs get project_id:: prefix."""
    result = mock_pipeline.index_file(
        Path("/proj/main.py"),
        project_id="eve",
        project_root=Path("/proj"),
    )
    # Verify nodes have prefixed IDs
    node_calls = mock_pipeline.graph.add_node.call_args_list
    for call in node_calls:
        node = call[0][0]
        assert node.id.startswith("eve::"), f"Node ID missing prefix: {node.id}"

def test_index_file_uses_relative_paths(mock_pipeline):
    """File paths stored in graph are relative to project_root."""
    result = mock_pipeline.index_file(
        Path("/proj/services/main.py"),
        project_id="eve",
        project_root=Path("/proj"),
    )
    node_calls = mock_pipeline.graph.add_node.call_args_list
    for call in node_calls:
        node = call[0][0]
        file_prop = node.properties.get("file") or node.properties.get("path")
        if file_prop:
            assert not file_prop.startswith("/"), f"Path should be relative: {file_prop}"
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_indexer/test_pipeline.py -v -k "project_id or relative"`
Expected: FAIL

**Step 3: Modify pipeline.py**

`index_file()` signature change:
```python
def index_file(self, path: Path, project_id: str = "", project_root: Path | None = None) -> IndexResult:
```

Inside `index_file()`:
- After parsing, prefix all node IDs: `node.id = f"{project_id}::{node.id}"` (via helper)
- Convert absolute paths to relative: `rel_path = str(path.relative_to(project_root))` if project_root given
- Store `rel_path` in node properties instead of absolute path
- Pass `project_id` to `vector_store.add()`
- Chunk IDs also get prefix (they derive from node IDs)

Add helper function:
```python
def _prefix_id(project_id: str, original_id: str) -> str:
    """Add project prefix to a node/edge ID."""
    if project_id and not original_id.startswith(f"{project_id}::"):
        return f"{project_id}::{original_id}"
    return original_id
```

`index_project()` and `update_project()` get `project_id` parameter too.

**Step 4: Run all pipeline tests**

Run: `.venv/bin/python -m pytest tests/test_indexer/test_pipeline.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add nemesis/indexer/pipeline.py nemesis/indexer/chunker.py tests/test_indexer/test_pipeline.py
git commit -m "feat: add project_id prefix and relative paths to pipeline"
```

---

### Task 5: Engine — wire registry and central paths

**Files:**
- Modify: `nemesis/core/engine.py`
- Modify: `tests/test_core/test_engine.py`

**Step 1: Write the failing test**

```python
def test_engine_uses_central_data_dir(tmp_path, mock_config):
    mock_config.data_dir = tmp_path
    engine = NemesisEngine(mock_config)
    engine.initialize()
    # Graph should be at data_dir/graph
    assert str(tmp_path / "graph") in str(engine._graph._db_path)

def test_engine_has_registry(tmp_path, mock_config):
    mock_config.data_dir = tmp_path
    engine = NemesisEngine(mock_config)
    engine.initialize()
    assert engine.registry is not None
```

**Step 2: Run to verify failure**

**Step 3: Modify engine.py**

Change `initialize()` to use `config.graph_dir` and `config.vector_dir` instead of `config.project_root / config.graph_path`. Add `self._registry` initialization.

```python
def initialize(self) -> None:
    if self._initialized:
        return
    cfg = self.config

    # Graph — central location
    graph_kwargs = {"db_path": str(cfg.graph_dir)}
    if cfg.graph_backend == "neo4j":
        graph_kwargs = {"uri": cfg.neo4j_uri, "user": cfg.neo4j_user, "password": cfg.neo4j_password}
    self._graph = create_graph_adapter(backend=cfg.graph_backend, create_schema=True, **graph_kwargs)

    # Vector Store — central location
    raw_store = VectorStore(path=str(cfg.vector_dir))
    self._vector_store = SyncVectorStoreWrapper(raw_store)

    # Registry
    self._registry = ProjectRegistry(cfg.registry_path)

    # ... rest stays the same
```

**Step 4: Run engine tests**

Run: `.venv/bin/python -m pytest tests/test_core/test_engine.py -v`

**Step 5: Commit**

```bash
git add nemesis/core/engine.py tests/test_core/test_engine.py
git commit -m "refactor: engine uses central data_dir and registry"
```

---

### Task 6: Tools — add project parameter

**Files:**
- Modify: `nemesis/tools/tools.py`
- Modify: `tests/test_tools/test_tools.py`

**Step 1: Write the failing tests**

```python
def test_search_code_with_project_filter(mock_engine):
    result = search_code(mock_engine, query="hello", project="eve")
    # vector_store.search should have been called with project_id="eve"
    mock_engine.vector_store.search.assert_called_once()
    call_kwargs = mock_engine.vector_store.search.call_args
    assert call_kwargs.kwargs.get("project_id") == "eve" or call_kwargs[1].get("project_id") == "eve"

def test_search_code_cross_project(mock_engine):
    result = search_code(mock_engine, query="hello")
    call_kwargs = mock_engine.vector_store.search.call_args
    assert call_kwargs.kwargs.get("project_id") is None or "project_id" not in call_kwargs.kwargs

def test_index_project_registers_project(mock_engine):
    result = index_project(mock_engine, path="/tmp/myproj", languages=["python"])
    mock_engine.registry.register.assert_called_once()

def test_list_projects(mock_engine):
    result = list_projects(mock_engine)
    assert "projects" in result

def test_remove_project(mock_engine):
    result = remove_project(mock_engine, name="eve")
    mock_engine.registry.unregister.assert_called_with("eve")
```

**Step 2: Run to verify failure**

**Step 3: Modify tools.py**

- `search_code()` — add `project: str | None = None`, pass to `vector_store.search(project_id=project)`
- `get_context()` — resolve project from file_path via `engine.registry.resolve()`, filter graph results
- `index_project()` — register project in registry, pass `project_id` to pipeline
- `update_project()` — resolve project, pass `project_id`
- Add `list_projects(engine)` function
- Add `remove_project(engine, name)` function — unregister + delete from graph/vector

**Step 4: Run tool tests**

Run: `.venv/bin/python -m pytest tests/test_tools/test_tools.py -v`

**Step 5: Commit**

```bash
git add nemesis/tools/tools.py tests/test_tools/test_tools.py
git commit -m "feat: tools support project parameter + list/remove"
```

---

### Task 7: MCP Server — new tool definitions

**Files:**
- Modify: `nemesis/core/server.py`
- Modify: `tests/test_core/test_server.py`

**Step 1: Write the failing test**

```python
def test_tool_definitions_include_project_param():
    from nemesis.core.server import TOOL_DEFINITIONS
    search_tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "search_code")
    assert "project" in search_tool["inputSchema"]["properties"]

def test_tool_definitions_include_list_projects():
    from nemesis.core.server import TOOL_DEFINITIONS
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert "list_projects" in names
    assert "remove_project" in names
```

**Step 2: Run to verify failure**

**Step 3: Modify server.py**

- Add `"project"` optional field to `search_code`, `get_context`, `index_project`, `update_project` tool schemas
- Add `"name"` optional field to `index_project` schema
- Add `list_projects` and `remove_project` tool definitions
- Add to `_TOOL_DISPATCH`
- Remove `NEMESIS_PROJECT_ROOT` dependency from `run_stdio_server()`

**Step 4: Run server tests**

Run: `.venv/bin/python -m pytest tests/test_core/test_server.py -v`

**Step 5: Commit**

```bash
git add nemesis/core/server.py tests/test_core/test_server.py
git commit -m "feat: MCP server supports multi-project tools"
```

---

### Task 8: CLI — update commands

**Files:**
- Modify: `nemesis/core/cli.py`
- Modify: `tests/test_core/test_cli.py`

**Step 1: Write the failing tests**

```python
def test_index_command_with_name(runner):
    result = runner.invoke(main, ["index", "/tmp/proj", "-l", "python", "--name", "eve"])
    assert result.exit_code == 0

def test_query_with_project_filter(runner):
    result = runner.invoke(main, ["query", "hello", "--project", "eve"])
    # Should not crash

def test_projects_command(runner):
    result = runner.invoke(main, ["projects"])
    assert result.exit_code == 0

def test_remove_command(runner):
    result = runner.invoke(main, ["remove", "eve"])
    # May fail with "not found" but should not crash
```

**Step 2: Run to verify failure**

**Step 3: Modify cli.py**

- `index` command: remove `--project-root`, add `--name`. Use `engine.registry.register()` before indexing. Pass `project_id` to pipeline.
- `query` command: remove `--project-root`, add `--project`. Pass to `search()`.
- Add `projects` command: lists all registered projects.
- Add `remove` command: unregisters a project and deletes its data.
- `serve` command: no changes needed (already simple).
- `watch` command: add `--name` for project tracking.

**Step 4: Run CLI tests**

Run: `.venv/bin/python -m pytest tests/test_core/test_cli.py -v`

**Step 5: Commit**

```bash
git add nemesis/core/cli.py tests/test_core/test_cli.py
git commit -m "feat: CLI supports multi-project commands"
```

---

### Task 9: Fix remaining tests

**Files:**
- Modify: various test files that reference `project_root` or old config

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`

**Step 2: Fix each failing test**

Most failures will be:
- Tests creating `NemesisConfig(project_root=...)` — change to `NemesisConfig(data_dir=tmp_path)`
- Tests expecting `config.project_root` — remove or update
- Integration tests using old path structure
- Mock setups that expect old Engine interface

**Step 3: Run full suite again**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all 431+ tests PASS

**Step 4: Run Rust tests**

Run: `cargo test --quiet`
Expected: 36 PASS (Rust parser is unchanged)

**Step 5: Commit**

```bash
git add -u
git commit -m "fix: update tests for multi-project architecture"
```

---

### Task 10: Update .claude.json and re-index

**Step 1: Update MCP server config**

Remove `NEMESIS_PROJECT_ROOT` from `~/.claude.json`:

```json
"nemesis": {
  "type": "stdio",
  "command": "/home/andreas/projects/nemesis/.venv/bin/nemesis",
  "args": ["serve"],
  "env": {
    "NEMESIS_OPENAI_API_KEY": "sk-..."
  }
}
```

**Step 2: Create central data directory**

```bash
mkdir -p ~/.nemesis
```

**Step 3: Index both projects**

```bash
nemesis index /home/andreas/projects/nemesis --name nemesis -l python
nemesis index /home/andreas/projects/Eve-Online-Copilot --name eve -l python -l typescript
```

**Step 4: Verify**

```bash
nemesis projects
nemesis query "MCP server" --project nemesis
nemesis query "ESI client" --project eve
nemesis query "config"  # cross-project
```

**Step 5: Clean up old data**

```bash
rm -rf /home/andreas/projects/nemesis/.nemesis
rm -rf /home/andreas/projects/Eve-Online-Copilot/.nemesis
```

**Step 6: Commit config changes**

```bash
git add -u
git commit -m "chore: remove old per-project .nemesis directories"
```
