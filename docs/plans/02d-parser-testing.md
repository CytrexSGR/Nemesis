# Rust Parser — Arbeitspaket B4: Integration Tests + Lint

> **Arbeitspaket B4** — Teil 4 von 4 des Rust Parser Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Multi-Language-Integrationstests durch die Python-Bridge und abschliessende Ruff/Clippy-Konformitaet sicherstellen. (Tasks 11, 12)

**Tech Stack:** Python 3.12, pytest, ruff, cargo clippy, nemesis_parse (Rust Extension)

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [02c-python-bridge.md](02c-python-bridge.md) (Arbeitspaket B3)

---

## Task 11: Multi-Language Integration Tests

End-to-end tests that parse real files in all three languages through the bridge.

**Files:**
- Create: `tests/test_parser/test_integration.py`

### Step 1 — Write the integration tests

```python
# tests/test_parser/test_integration.py
"""Integration tests for multi-language parsing through the Python bridge."""

from pathlib import Path

import pytest

from nemesis.parser import ParserBridge, NodeKind, EdgeKind, ExtractionResult


@pytest.fixture
def bridge() -> ParserBridge:
    return ParserBridge()


class TestPythonIntegration:
    """Full parsing integration for Python files."""

    def test_parse_complex_python_file(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        assert result.language == "python"
        assert result.node_count > 5

    def test_extracts_classes_with_inheritance(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        classes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        class_names = {c.name for c in classes}
        assert "BaseService" in class_names
        assert "UserService" in class_names

    def test_extracts_inherits_edge(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        inherits = [e for e in result.edges if e.kind == EdgeKind.INHERITS]
        assert len(inherits) >= 1
        # UserService inherits BaseService
        assert any("UserService" in e.source_id for e in inherits)

    def test_extracts_imports(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        imports = [n for n in result.nodes if n.kind == NodeKind.IMPORT]
        assert len(imports) >= 2

    def test_extracts_methods_with_visibility(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_map = {m.name: m for m in methods}
        assert "get_user" in method_map
        assert method_map["get_user"].visibility == "public"
        assert "_internal_method" in method_map
        assert method_map["_internal_method"].visibility == "protected"

    def test_extracts_standalone_functions(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        func_names = {f.name for f in funcs}
        assert "create_service" in func_names


class TestTypeScriptIntegration:
    """Full parsing integration for TypeScript files."""

    def test_parse_typescript_file(
        self, bridge: ParserBridge, sample_typescript_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_typescript_file))
        assert result.language == "typescript"
        assert result.node_count > 3

    def test_extracts_interface(
        self, bridge: ParserBridge, sample_typescript_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_typescript_file))
        ifaces = [n for n in result.nodes if n.kind == NodeKind.INTERFACE]
        assert len(ifaces) >= 1
        assert any(i.name == "User" for i in ifaces)

    def test_extracts_class_with_methods(
        self, bridge: ParserBridge, sample_typescript_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_typescript_file))
        classes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        assert any(c.name == "UserService" for c in classes)
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "getUser" in method_names
        assert "addUser" in method_names

    def test_extracts_function(
        self, bridge: ParserBridge, sample_typescript_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_typescript_file))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert any(f.name == "createApp" for f in funcs)


class TestRustIntegration:
    """Full parsing integration for Rust files."""

    def test_parse_rust_file(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        assert result.language == "rust"
        assert result.node_count > 3

    def test_extracts_struct_as_class(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        structs = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        assert any(s.name == "Person" for s in structs)

    def test_extracts_trait_as_interface(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        traits = [n for n in result.nodes if n.kind == NodeKind.INTERFACE]
        assert any(t.name == "Greetable" for t in traits)

    def test_extracts_impl_methods(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "new" in method_names
        assert "is_adult" in method_names

    def test_extracts_pub_function(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        create_fn = next((f for f in funcs if f.name == "create_person"), None)
        assert create_fn is not None
        assert create_fn.visibility == "pub"

    def test_has_method_edges(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        hm = [e for e in result.edges if e.kind == EdgeKind.HAS_METHOD]
        assert len(hm) >= 2  # new, is_adult from impl Person
```

### Step 2 — Run tests

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_integration.py -v`
Expected: PASS (~17 tests)

Hinweis: Benoetigt kompilierte Rust-Extension (`maturin develop --release`).

### Step 3 — Commit

```bash
git add tests/test_parser/test_integration.py
git commit -m "test(parser): add multi-language integration tests for Python, TypeScript, and Rust"
```

---

## Task 12: Ruff-Konformitaet und finaler Gesamttest

Linting, Formatting, und finaler Testlauf ueber alle Parser-Tests.

**Files:**
- Modify: alle bestehenden `.py` Dateien im `nemesis/parser/` und `tests/test_parser/` (falls noetig)

### Step 1 — Ruff Check

Run: `cd /home/andreas/projects/nemesis && python3 -m ruff check nemesis/parser/ tests/test_parser/`
Expected: `All checks passed!`

### Step 2 — Ruff Format

Run: `cd /home/andreas/projects/nemesis && python3 -m ruff format nemesis/parser/ tests/test_parser/`

### Step 3 — Cargo clippy

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo clippy -- -D warnings`
Expected: No warnings

### Step 4 — Full Rust test suite

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo test`
Expected: All Rust tests pass (~36 tests)

### Step 5 — Full Python test suite

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/ -v --tb=short`
Expected: All Python tests pass (~51 tests)

### Step 6 — Commit

```bash
git add -A
git commit -m "style(parse): apply ruff formatting and clippy fixes to parser module"
```

---

## Zusammenfassung B4

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 11 | Multi-Language Integration Tests | `tests/test_parser/test_integration.py` | 17 Python-Tests |
| 12 | Ruff + Clippy + Gesamttest | Bestehende Dateien | Lint + ~88 Tests gesamt |

**Gesamt B4: 2 Tasks, 17 neue Python-Tests, 2 Commits**

---

## Gesamtuebersicht aller Arbeitspakete

| Paket | Tasks | Beschreibung | Tests |
|-------|-------|-------------|-------|
| [B1](02a-rust-core-models.md) | 1, 2, 3 | Rust Data Models + Grammars + Parser | 16 Rust-Tests |
| [B2](02b-rust-extraction.md) | 4, 5, 6 | Extractor + PyO3 Bindings + Build | 20 Rust-Tests |
| [B3](02c-python-bridge.md) | 7, 8, 9, 10 | Python Bridge + Public API + Fixtures | 34 Python-Tests |
| [B4](02d-parser-testing.md) | 11, 12 | Integration Tests + Lint | 17 Python-Tests |

**Gesamt: 12 Tasks, ~36 Rust-Tests + ~51 Python-Tests = ~87 Tests, 12 Commits**

---

**Navigation:**
- Vorheriges Paket: [B3 — Python Bridge](02c-python-bridge.md)
- Gesamtplan: [02-rust-parser.md](02-rust-parser.md)
