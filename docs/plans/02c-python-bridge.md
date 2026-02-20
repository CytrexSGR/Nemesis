# Rust Parser — Arbeitspaket B3: Python Bridge + Public API + Fixtures

> **Arbeitspaket B3** — Teil 3 von 4 des Rust Parser Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Python-seitige Dataclasses, den Python-Bridge-Wrapper, das oeffentliche Parser-API und Test-Fixtures implementieren. (Tasks 7, 8, 9, 10)

**Tech Stack:** Python 3.12, dataclasses, pytest, nemesis_parse (Rust Extension)

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [02b-rust-extraction.md](02b-rust-extraction.md) (Arbeitspaket B2)

---

## Task 7: Python Bridge — Data Classes (nemesis/parser/models.py)

Python-side dataclasses mirroring the Rust models, used by the bridge.

**Files:**
- Create: `nemesis/parser/models.py`
- Create: `tests/test_parser/__init__.py`
- Create: `tests/test_parser/test_models.py`

### Step 1 — Write the failing test

```python
# tests/test_parser/__init__.py
```

```python
# tests/test_parser/test_models.py
"""Tests for parser Python data models."""

from nemesis.parser.models import CodeNode, CodeEdge, ExtractionResult, NodeKind, EdgeKind


class TestNodeKind:
    def test_all_kinds_exist(self) -> None:
        kinds = [NodeKind.FILE, NodeKind.MODULE, NodeKind.CLASS, NodeKind.FUNCTION,
                 NodeKind.METHOD, NodeKind.INTERFACE, NodeKind.VARIABLE, NodeKind.IMPORT]
        assert len(kinds) == 8

    def test_kind_values(self) -> None:
        assert NodeKind.FILE == "File"
        assert NodeKind.FUNCTION == "Function"


class TestEdgeKind:
    def test_all_kinds_exist(self) -> None:
        kinds = [EdgeKind.CONTAINS, EdgeKind.HAS_METHOD, EdgeKind.INHERITS,
                 EdgeKind.IMPLEMENTS, EdgeKind.CALLS, EdgeKind.IMPORTS,
                 EdgeKind.RETURNS, EdgeKind.ACCEPTS]
        assert len(kinds) == 8

    def test_kind_values(self) -> None:
        assert EdgeKind.CONTAINS == "CONTAINS"
        assert EdgeKind.HAS_METHOD == "HAS_METHOD"


class TestCodeNode:
    def test_creation(self) -> None:
        node = CodeNode(
            id="func:test.py:hello:1",
            kind=NodeKind.FUNCTION,
            name="hello",
            file="test.py",
            line_start=1,
            line_end=3,
            language="python",
        )
        assert node.name == "hello"
        assert node.kind == NodeKind.FUNCTION
        assert node.docstring is None
        assert node.is_async is False

    def test_from_dict(self) -> None:
        data = {
            "id": "class:t.py:Foo:1",
            "kind": "Class",
            "name": "Foo",
            "file": "t.py",
            "line_start": 1,
            "line_end": 5,
            "language": "python",
            "docstring": "A foo.",
            "is_async": False,
        }
        node = CodeNode.from_dict(data)
        assert node.name == "Foo"
        assert node.docstring == "A foo."


class TestCodeEdge:
    def test_creation(self) -> None:
        edge = CodeEdge(
            source_id="file:test.py",
            target_id="func:test.py:hello:1",
            kind=EdgeKind.CONTAINS,
            file="test.py",
        )
        assert edge.kind == EdgeKind.CONTAINS

    def test_from_dict(self) -> None:
        data = {
            "source_id": "a",
            "target_id": "b",
            "kind": "CONTAINS",
            "file": "t.py",
        }
        edge = CodeEdge.from_dict(data)
        assert edge.kind == EdgeKind.CONTAINS


class TestExtractionResult:
    def test_creation(self) -> None:
        result = ExtractionResult(
            file="test.py",
            language="python",
            nodes=[],
            edges=[],
        )
        assert result.file == "test.py"
        assert len(result.nodes) == 0

    def test_from_dict(self) -> None:
        data = {
            "file": "t.py",
            "language": "python",
            "nodes": [
                {"id": "file:t.py", "kind": "File", "name": "t.py",
                 "file": "t.py", "line_start": 1, "line_end": 1,
                 "language": "python", "is_async": False},
            ],
            "edges": [],
        }
        result = ExtractionResult.from_dict(data)
        assert len(result.nodes) == 1
        assert result.nodes[0].kind == NodeKind.FILE

    def test_node_count_property(self) -> None:
        node = CodeNode(id="x", kind=NodeKind.FILE, name="x", file="x",
                        line_start=1, line_end=1, language="python")
        result = ExtractionResult(file="x", language="python",
                                  nodes=[node], edges=[])
        assert result.node_count == 1
        assert result.edge_count == 0
```

### Step 2 — Run test to verify it fails

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_models.py -v`
Expected: FAIL with ImportError

### Step 3 — Write implementation

```python
# nemesis/parser/models.py
"""Python data models mirroring the Rust nemesis-parse structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class NodeKind:
    """Code node type constants."""

    FILE = "File"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    INTERFACE = "Interface"
    VARIABLE = "Variable"
    IMPORT = "Import"


class EdgeKind:
    """Code edge type constants."""

    CONTAINS = "CONTAINS"
    HAS_METHOD = "HAS_METHOD"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    RETURNS = "RETURNS"
    ACCEPTS = "ACCEPTS"


@dataclass
class CodeNode:
    """A structured code node extracted from an AST."""

    id: str
    kind: str
    name: str
    file: str
    line_start: int
    line_end: int
    language: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    type_hint: Optional[str] = None
    scope: Optional[str] = None
    source: Optional[str] = None
    alias: Optional[str] = None
    visibility: Optional[str] = None
    is_async: bool = False
    parent_class: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> CodeNode:
        """Create a CodeNode from a dictionary (e.g. from JSON)."""
        return cls(
            id=data["id"],
            kind=data["kind"],
            name=data["name"],
            file=data["file"],
            line_start=data["line_start"],
            line_end=data["line_end"],
            language=data["language"],
            docstring=data.get("docstring"),
            signature=data.get("signature"),
            type_hint=data.get("type_hint"),
            scope=data.get("scope"),
            source=data.get("source"),
            alias=data.get("alias"),
            visibility=data.get("visibility"),
            is_async=data.get("is_async", False),
            parent_class=data.get("parent_class"),
        )


@dataclass
class CodeEdge:
    """A directed edge between two code nodes."""

    source_id: str
    target_id: str
    kind: str
    file: str

    @classmethod
    def from_dict(cls, data: dict) -> CodeEdge:
        """Create a CodeEdge from a dictionary (e.g. from JSON)."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            kind=data["kind"],
            file=data["file"],
        )


@dataclass
class ExtractionResult:
    """Complete extraction result for a single file."""

    file: str
    language: str
    nodes: list[CodeNode] = field(default_factory=list)
    edges: list[CodeEdge] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @classmethod
    def from_dict(cls, data: dict) -> ExtractionResult:
        """Create an ExtractionResult from a dictionary (e.g. from JSON)."""
        nodes = [CodeNode.from_dict(n) for n in data.get("nodes", [])]
        edges = [CodeEdge.from_dict(e) for e in data.get("edges", [])]
        return cls(
            file=data["file"],
            language=data["language"],
            nodes=nodes,
            edges=edges,
        )
```

### Step 4 — Run test to verify pass

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_models.py -v`
Expected: PASS (12 tests)

### Step 5 — Commit

```bash
git add nemesis/parser/models.py tests/test_parser/
git commit -m "feat(parser): add Python data models for code nodes, edges, and extraction results"
```

---

## Task 8: Python Bridge (nemesis/parser/bridge.py)

Python wrapper around the Rust extension with error handling, caching, and Pythonic API.

**Files:**
- Create: `nemesis/parser/bridge.py`
- Create: `tests/test_parser/test_bridge.py`

### Step 1 — Write the failing test

```python
# tests/test_parser/test_bridge.py
"""Tests for the Python bridge to nemesis-parse Rust extension."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemesis.parser.bridge import ParserBridge, ParserError
from nemesis.parser.models import CodeNode, ExtractionResult, NodeKind, EdgeKind


class TestParserBridgeInit:
    def test_bridge_creation(self) -> None:
        bridge = ParserBridge()
        assert bridge is not None

    def test_bridge_has_native_module(self) -> None:
        bridge = ParserBridge()
        assert bridge.native_available is True or bridge.native_available is False

    def test_supported_languages(self) -> None:
        bridge = ParserBridge()
        langs = bridge.supported_languages()
        assert "python" in langs


class TestParserBridgeDetect:
    def test_detect_python(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("app.py") == "python"

    def test_detect_typescript(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("app.ts") == "typescript"

    def test_detect_rust(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("main.rs") == "rust"

    def test_detect_unsupported(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.detect_language("file.java")


class TestParserBridgeParseString:
    def test_parse_python_string(self) -> None:
        bridge = ParserBridge()
        result = bridge.parse_string("def hello(): pass\n", "python", "test.py")
        assert isinstance(result, ExtractionResult)
        assert result.file == "test.py"
        assert result.language == "python"
        assert result.node_count >= 2  # File + Function

    def test_parse_python_class(self) -> None:
        bridge = ParserBridge()
        src = "class Foo:\n    def bar(self):\n        pass\n"
        result = bridge.parse_string(src, "python", "test.py")
        class_nodes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        method_nodes = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        assert len(class_nodes) == 1
        assert len(method_nodes) == 1
        assert class_nodes[0].name == "Foo"

    def test_parse_typescript_string(self) -> None:
        bridge = ParserBridge()
        src = "function greet(name: string): void { console.log(name); }\n"
        result = bridge.parse_string(src, "typescript", "test.ts")
        assert result.language == "typescript"
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 1

    def test_parse_rust_string(self) -> None:
        bridge = ParserBridge()
        src = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        result = bridge.parse_string(src, "rust", "lib.rs")
        assert result.language == "rust"
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 1

    def test_parse_empty_source(self) -> None:
        bridge = ParserBridge()
        result = bridge.parse_string("", "python", "empty.py")
        assert result.node_count >= 1  # At least File node

    def test_parse_invalid_language_raises(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.parse_string("code", "java", "test.java")


class TestParserBridgeParseFile:
    def test_parse_python_file(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        assert result.language == "python"
        class_nodes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        assert len(class_nodes) >= 1
        assert any(n.name == "Calculator" for n in class_nodes)

    def test_parse_nonexistent_file_raises(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.parse_file("/nonexistent/path/file.py")

    def test_parse_file_extracts_methods(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "add" in method_names
        assert "subtract" in method_names

    def test_parse_file_extracts_edges(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        contains = [e for e in result.edges if e.kind == EdgeKind.CONTAINS]
        has_method = [e for e in result.edges if e.kind == EdgeKind.HAS_METHOD]
        assert len(contains) >= 1
        assert len(has_method) >= 1


class TestParserBridgeEdgeCases:
    def test_parse_syntax_error_still_produces_partial(self) -> None:
        """Tree-sitter is error-tolerant; partial results are expected."""
        bridge = ParserBridge()
        src = "def broken(:\n    pass\n"
        result = bridge.parse_string(src, "python", "broken.py")
        # Should still produce a File node at minimum
        assert result.node_count >= 1

    def test_parse_large_source(self) -> None:
        bridge = ParserBridge()
        # Generate a file with 100 functions
        lines = []
        for i in range(100):
            lines.append(f"def func_{i}():\n    pass\n")
        src = "\n".join(lines)
        result = bridge.parse_string(src, "python", "big.py")
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 100
```

### Step 2 — Run test to verify it fails

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_bridge.py -v`
Expected: FAIL with ImportError

### Step 3 — Write implementation

```python
# nemesis/parser/bridge.py
"""Python bridge to the nemesis-parse Rust extension.

Provides a Pythonic API around the native Tree-sitter parser.
Falls back to helpful error messages if the extension is not compiled.
"""

from __future__ import annotations

import json
from pathlib import Path

from nemesis.parser.models import CodeEdge, CodeNode, ExtractionResult


class ParserError(Exception):
    """Raised when parsing or extraction fails."""


# Try to import the native Rust extension
_native = None
try:
    import nemesis_parse as _native  # type: ignore[import-not-found]
except ImportError:
    _native = None


class ParserBridge:
    """High-level Python interface to the nemesis-parse Rust extension.

    Usage::

        bridge = ParserBridge()
        result = bridge.parse_file("src/main.py")
        for node in result.nodes:
            print(f"{node.kind}: {node.name}")
    """

    def __init__(self) -> None:
        self._native = _native

    @property
    def native_available(self) -> bool:
        """Whether the Rust native extension is available."""
        return self._native is not None

    def _require_native(self) -> None:
        if self._native is None:
            raise ParserError(
                "nemesis-parse native extension not available. "
                "Build it with: cd nemesis-parse && maturin develop --release"
            )

    def supported_languages(self) -> list[str]:
        """Return the list of supported programming languages."""
        if self._native is not None:
            return self._native.supported_languages()
        return ["python", "typescript", "tsx", "rust"]

    def detect_language(self, file_path: str) -> str:
        """Detect the programming language from a file extension.

        Args:
            file_path: Path to the file (only extension is used).

        Returns:
            Language identifier string.

        Raises:
            ParserError: If the file extension is not supported.
        """
        if self._native is not None:
            try:
                return self._native.detect_language(file_path)
            except ValueError as e:
                raise ParserError(str(e)) from e

        # Fallback detection
        ext = Path(file_path).suffix.lstrip(".")
        mapping = {"py": "python", "ts": "typescript", "tsx": "tsx", "rs": "rust"}
        if ext not in mapping:
            raise ParserError(f"Unsupported file extension: {ext}")
        return mapping[ext]

    def parse_string(
        self,
        source: str,
        language: str,
        file_path: str,
    ) -> ExtractionResult:
        """Parse a source string and extract code nodes and edges.

        Args:
            source: The source code string.
            language: Language identifier (python, typescript, tsx, rust).
            file_path: Virtual file path for node IDs.

        Returns:
            ExtractionResult with nodes and edges.

        Raises:
            ParserError: If parsing fails or language is unsupported.
        """
        self._require_native()
        try:
            json_str = self._native.parse_string(source, language, file_path)
            data = json.loads(json_str)
            return ExtractionResult.from_dict(data)
        except (ValueError, json.JSONDecodeError) as e:
            raise ParserError(f"Parse failed: {e}") from e

    def parse_file(self, file_path: str) -> ExtractionResult:
        """Parse a file from disk, auto-detecting the language.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            ExtractionResult with nodes and edges.

        Raises:
            ParserError: If the file doesn't exist, can't be read, or language
                        is unsupported.
        """
        self._require_native()
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"File not found: {file_path}")

        try:
            json_str = self._native.parse_file(str(path))
            data = json.loads(json_str)
            return ExtractionResult.from_dict(data)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            raise ParserError(f"Parse failed for {file_path}: {e}") from e

    def extract_nodes(
        self,
        source: str,
        language: str,
        file_path: str,
    ) -> list[CodeNode]:
        """Extract only code nodes from a source string.

        Args:
            source: The source code string.
            language: Language identifier.
            file_path: Virtual file path for node IDs.

        Returns:
            List of CodeNode objects.
        """
        self._require_native()
        try:
            json_str = self._native.extract_nodes(source, language, file_path)
            data = json.loads(json_str)
            return [CodeNode.from_dict(n) for n in data]
        except (ValueError, json.JSONDecodeError) as e:
            raise ParserError(f"Node extraction failed: {e}") from e

    def extract_edges(
        self,
        source: str,
        language: str,
        file_path: str,
    ) -> list[CodeEdge]:
        """Extract only code edges from a source string.

        Args:
            source: The source code string.
            language: Language identifier.
            file_path: Virtual file path for node IDs.

        Returns:
            List of CodeEdge objects.
        """
        self._require_native()
        try:
            json_str = self._native.extract_edges(source, language, file_path)
            data = json.loads(json_str)
            return [CodeEdge.from_dict(e) for e in data]
        except (ValueError, json.JSONDecodeError) as e:
            raise ParserError(f"Edge extraction failed: {e}") from e
```

### Step 4 — Run test to verify pass

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_bridge.py -v`
Expected: PASS (18 tests)

Hinweis: Die Tests benoetigen die kompilierte Rust-Extension. Vor dem Test: `cd nemesis-parse && maturin develop --release`

### Step 5 — Commit

```bash
git add nemesis/parser/bridge.py tests/test_parser/test_bridge.py
git commit -m "feat(parser): add Python bridge with parse_file, parse_string, extract_nodes, extract_edges"
```

---

## Task 9: Update parser __init__.py with public API

Clean public exports from the parser subpackage.

**Files:**
- Modify: `nemesis/parser/__init__.py`
- Create: `tests/test_parser/test_parser_init.py`

### Step 1 — Write the failing test

```python
# tests/test_parser/test_parser_init.py
"""Tests for parser subpackage public API."""


def test_parser_exports_bridge() -> None:
    from nemesis.parser import ParserBridge
    assert ParserBridge is not None


def test_parser_exports_error() -> None:
    from nemesis.parser import ParserError
    assert ParserError is not None


def test_parser_exports_models() -> None:
    from nemesis.parser import CodeNode, CodeEdge, ExtractionResult
    assert CodeNode is not None
    assert CodeEdge is not None
    assert ExtractionResult is not None


def test_parser_exports_kinds() -> None:
    from nemesis.parser import NodeKind, EdgeKind
    assert NodeKind.FILE == "File"
    assert EdgeKind.CONTAINS == "CONTAINS"
```

### Step 2 — Run test to verify it fails

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_parser_init.py -v`
Expected: FAIL with ImportError

### Step 3 — Write implementation

```python
# nemesis/parser/__init__.py
"""Parser module — PyO3 bridge to nemesis-parse Rust crate.

Public API::

    from nemesis.parser import ParserBridge, ParserError
    from nemesis.parser import CodeNode, CodeEdge, ExtractionResult
    from nemesis.parser import NodeKind, EdgeKind
"""

from nemesis.parser.bridge import ParserBridge, ParserError
from nemesis.parser.models import (
    CodeEdge,
    CodeNode,
    EdgeKind,
    ExtractionResult,
    NodeKind,
)

__all__ = [
    "ParserBridge",
    "ParserError",
    "CodeNode",
    "CodeEdge",
    "ExtractionResult",
    "NodeKind",
    "EdgeKind",
]
```

### Step 4 — Run test to verify pass

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_parser_init.py -v`
Expected: PASS (4 tests)

### Step 5 — Commit

```bash
git add nemesis/parser/__init__.py tests/test_parser/test_parser_init.py
git commit -m "feat(parser): export clean public API from parser subpackage"
```

---

## Task 10: Test Fixtures for Parser Tests

Add shared fixtures (conftest) for parser test directory.

**Files:**
- Create: `tests/test_parser/conftest.py`

### Step 1 — Write the fixtures

```python
# tests/test_parser/conftest.py
"""Shared fixtures for parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_typescript_file(tmp_project: Path) -> Path:
    """Create a sample TypeScript file for testing."""
    code = '''import { Request, Response } from "express";

interface User {
  id: number;
  name: string;
}

class UserService {
  private users: User[] = [];

  getUser(id: number): User | undefined {
    return this.users.find(u => u.id === id);
  }

  addUser(user: User): void {
    this.users.push(user);
  }
}

function createApp(): void {
  const service = new UserService();
  console.log("App started");
}
'''
    file_path = tmp_project / "app.ts"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_rust_file(tmp_project: Path) -> Path:
    """Create a sample Rust file for testing."""
    code = '''use std::fmt;

pub trait Greetable {
    fn greet(&self) -> String;
}

pub struct Person {
    pub name: String,
    age: u32,
}

impl Person {
    pub fn new(name: &str, age: u32) -> Self {
        Person {
            name: name.to_string(),
            age,
        }
    }

    fn is_adult(&self) -> bool {
        self.age >= 18
    }
}

impl Greetable for Person {
    fn greet(&self) -> String {
        format!("Hello, I am {}", self.name)
    }
}

pub fn create_person(name: &str) -> Person {
    Person::new(name, 30)
}
'''
    file_path = tmp_project / "person.rs"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_complex_python(tmp_project: Path) -> Path:
    """Create a complex Python file with inheritance and imports."""
    code = '''"""Complex module for testing."""

import os
from pathlib import Path
from typing import List, Optional

MAX_SIZE: int = 1024


class BaseService:
    """Base class for services."""

    def __init__(self, name: str):
        self.name = name

    def _internal_method(self):
        pass

    def __repr__(self):
        return f"BaseService({self.name})"


class UserService(BaseService):
    """Service for user operations."""

    def __init__(self, name: str, db_url: str):
        super().__init__(name)
        self.db_url = db_url

    def get_user(self, user_id: int) -> Optional[dict]:
        """Get a user by ID."""
        return None

    async def fetch_users(self) -> List[dict]:
        """Async fetch all users."""
        return []


def create_service(name: str = "default") -> UserService:
    """Factory function."""
    return UserService(name, "sqlite:///db.sqlite")
'''
    file_path = tmp_project / "services.py"
    file_path.write_text(code)
    return file_path
```

### Step 2 — Validate fixtures load

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/ --collect-only 2>&1 | tail -5`
Expected: Shows collected tests without errors

### Step 3 — Commit

```bash
git add tests/test_parser/conftest.py
git commit -m "test(parser): add TypeScript, Rust, and complex Python fixtures for parser tests"
```

---

## Zusammenfassung B3

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 7 | Python Data Models | `nemesis/parser/models.py` | 12 Python-Tests |
| 8 | Python Bridge (bridge.py) | `nemesis/parser/bridge.py` | 18 Python-Tests |
| 9 | Parser __init__.py | `nemesis/parser/__init__.py` | 4 Python-Tests |
| 10 | Test Fixtures | `tests/test_parser/conftest.py` | Fixture-Validierung |

**Gesamt B3: 4 Tasks, 34 Python-Tests, 4 Commits**

---

**Navigation:**
- Vorheriges Paket: [B2 — Rust Extraction](02b-rust-extraction.md)
- Naechstes Paket: [B4 — Parser Testing](02d-parser-testing.md)
- Gesamtplan: [02-rust-parser.md](02-rust-parser.md)
