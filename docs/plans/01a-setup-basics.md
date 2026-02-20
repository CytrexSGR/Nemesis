> **Arbeitspaket A1** — Teil 1 von 3 des Project Setup Plans

# Project Setup: Git + pyproject.toml + Package-Struktur

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the complete project scaffolding for Nemesis — pyproject.toml, Python package structure, and Git setup — so that all subsequent phases can build on a working foundation.

**Architecture:** Monorepo with a Python package (`nemesis/`) and a Rust crate (`nemesis-parse/`). The Python package uses maturin as build backend for the Rust extension. Embedded databases (Kuzu, LanceDB) keep deployment simple. Pydantic BaseSettings provides typed configuration with env var override support.

**Tech Stack:** Python 3.11+, Rust/PyO3/maturin, pytest, ruff, Click, Pydantic, GitHub Actions

**Dieses Paket enthält:** Task 1 (Git init), Task 2 (pyproject.toml), Task 3 (Package-Struktur)

---

### Task 1: Git Repository initialisieren

**Files:**
- Create: `.gitignore`

**Step 1: Repository und .gitignore erstellen**

```bash
cd /home/andreas/projects/nemesis
git init
```

**Step 2: .gitignore schreiben**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
*.egg
dist/
build/
*.whl
.eggs/
*.so

# Virtual environments
.venv/
venv/
ENV/

# Rust
nemesis-parse/target/
**/*.rs.bk

# IDEs
.idea/
.vscode/
*.swp
*.swo
*~

# Testing
.pytest_cache/
htmlcov/
.coverage
.coverage.*
coverage.xml

# Databases (local dev)
*.kuzu/
*.lance/
*.lancedb/

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Nemesis project data
.nemesis/
```

**Step 3: Validierung**
Run: `cd /home/andreas/projects/nemesis && cat .gitignore | head -5`
Expected: Die ersten 5 Zeilen der .gitignore

**Step 4: Commit**
```bash
git add .gitignore README.md docs/
git commit -m "chore: init repository with .gitignore and docs"
```

---

### Task 2: pyproject.toml mit Build-System und Dependencies

**Files:**
- Create: `pyproject.toml`

**Step 1: pyproject.toml schreiben**

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "nemesis-ai"
version = "0.1.0"
description = "GraphRAG context engine for AI coding agents"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [
    { name = "CytrexSGR" },
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Rust",
    "Topic :: Software Development :: Libraries",
]
dependencies = [
    "kuzu>=0.4",
    "lancedb>=0.6",
    "openai>=1.0",
    "click>=8.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "watchdog>=4.0",
    "mcp[cli]>=1.0",
]

[project.optional-dependencies]
neo4j = ["neo4j>=5.0"]
local-embeddings = ["sentence-transformers>=2.0"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov",
    "ruff>=0.4",
    "maturin>=1.5",
]

[project.scripts]
nemesis = "nemesis.core.cli:main"

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "nemesis._nemesis_parse"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
]

[tool.ruff.lint.isort]
known-first-party = ["nemesis"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests requiring external services",
]
```

**Step 2: Validierung**
Run: `cd /home/andreas/projects/nemesis && python3 -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['name'], d['project']['version'])"`
Expected: `nemesis-ai 0.1.0`

**Step 3: Weitere Validierung**
Run: `cd /home/andreas/projects/nemesis && python3 -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['tool']['ruff']['line-length'], d['tool']['ruff']['target-version'])"`
Expected: `100 py311`

**Step 4: Commit**
```bash
git add pyproject.toml
git commit -m "feat: add pyproject.toml with maturin build system and all dependencies"
```

---

### Task 3: Python Package Struktur — alle __init__.py

**Files:**
- Create: `nemesis/__init__.py`
- Create: `nemesis/core/__init__.py`
- Create: `nemesis/indexer/__init__.py`
- Create: `nemesis/parser/__init__.py`
- Create: `nemesis/graph/__init__.py`
- Create: `nemesis/vector/__init__.py`
- Create: `nemesis/memory/__init__.py`
- Create: `nemesis/tools/__init__.py`
- Test: `tests/test_package_structure.py`

**Step 1: Write the failing test**

```python
# tests/test_package_structure.py
"""Tests for package structure — all subpackages must be importable."""


def test_nemesis_package_exists():
    """The root nemesis package is importable and has a version."""
    import nemesis

    assert hasattr(nemesis, "__version__")
    assert nemesis.__version__ == "0.1.0"


def test_core_subpackage():
    """The core subpackage is importable."""
    import nemesis.core

    assert nemesis.core is not None


def test_indexer_subpackage():
    """The indexer subpackage is importable."""
    import nemesis.indexer

    assert nemesis.indexer is not None


def test_parser_subpackage():
    """The parser subpackage is importable."""
    import nemesis.parser

    assert nemesis.parser is not None


def test_graph_subpackage():
    """The graph subpackage is importable."""
    import nemesis.graph

    assert nemesis.graph is not None


def test_vector_subpackage():
    """The vector subpackage is importable."""
    import nemesis.vector

    assert nemesis.vector is not None


def test_memory_subpackage():
    """The memory subpackage is importable."""
    import nemesis.memory

    assert nemesis.memory is not None


def test_tools_subpackage():
    """The tools subpackage is importable."""
    import nemesis.tools

    assert nemesis.tools is not None


def test_all_subpackages_list():
    """All expected subpackages are present."""
    expected = {"core", "indexer", "parser", "graph", "vector", "memory", "tools"}
    import nemesis

    # Check that all subpackages are listed
    for pkg in expected:
        __import__(f"nemesis.{pkg}")
```

**Step 2: Run test to verify it fails**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_package_structure.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nemesis'"

**Step 3: Write minimal implementation**

```python
# nemesis/__init__.py
"""Nemesis — GraphRAG context engine for AI coding agents."""

__version__ = "0.1.0"
```

```python
# nemesis/core/__init__.py
"""Core module — server, config, CLI, watcher."""
```

```python
# nemesis/indexer/__init__.py
"""Indexer module — pipeline, chunker, delta updates."""
```

```python
# nemesis/parser/__init__.py
"""Parser module — PyO3 bridge to nemesis-parse Rust crate."""
```

```python
# nemesis/graph/__init__.py
"""Graph module — abstract adapter, Kuzu, Neo4j."""
```

```python
# nemesis/vector/__init__.py
"""Vector module — embeddings, LanceDB store."""
```

```python
# nemesis/memory/__init__.py
"""Memory module — rules, decisions, context, auto-learning."""
```

```python
# nemesis/tools/__init__.py
"""Tools module — MCP tool implementations."""
```

**Step 4: Run test to verify it passes**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_package_structure.py -v`
Expected: PASS (8 passed)

**Step 5: Commit**
```bash
git add nemesis/ tests/test_package_structure.py
git commit -m "feat: add Python package structure with all subpackages"
```

---

## Zusammenfassung

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 1 | Git init + .gitignore | `.gitignore` | — |
| 2 | pyproject.toml | `pyproject.toml` | Validierung |
| 3 | Python Package Struktur | 8x `__init__.py` | 8 Tests |

**Gesamt: 3 Tasks, ~8 Tests, 3 Commits**

---

**Vorheriges Paket:** —
**Nächstes Paket:** [01b — Core Infrastructure](01b-core-infrastructure.md)
