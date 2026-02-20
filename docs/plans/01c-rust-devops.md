> **Arbeitspaket A3** — Teil 3 von 3 des Project Setup Plans

# Project Setup: Rust Crate + GitHub Actions + Ruff

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up the Rust/PyO3 crate for tree-sitter parsing, GitHub Actions CI pipeline, and Ruff formatting — completing the project scaffolding for Nemesis.

**Architecture:** Monorepo with a Python package (`nemesis/`) and a Rust crate (`nemesis-parse/`). The Python package uses maturin as build backend for the Rust extension. Embedded databases (Kuzu, LanceDB) keep deployment simple. Pydantic BaseSettings provides typed configuration with env var override support.

**Tech Stack:** Python 3.11+, Rust/PyO3/maturin, pytest, ruff, Click, Pydantic, GitHub Actions

**Dieses Paket enthält:** Task 6 (Rust Crate), Task 8 (GitHub Actions), Task 9 (Ruff)

---

### Task 6: Rust Crate Scaffolding (nemesis-parse/)

**Files:**
- Create: `nemesis-parse/Cargo.toml`
- Create: `nemesis-parse/pyproject.toml`
- Create: `nemesis-parse/src/lib.rs`

**Step 1: Cargo.toml schreiben**

```toml
# nemesis-parse/Cargo.toml
[package]
name = "nemesis-parse"
version = "0.1.0"
edition = "2021"
description = "Tree-sitter AST parser for Nemesis — Python extension via PyO3"

[lib]
name = "nemesis_parse"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
tree-sitter = "0.24"
tree-sitter-python = "0.23"
tree-sitter-typescript = "0.23"
tree-sitter-rust = "0.23"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Step 2: nemesis-parse/pyproject.toml schreiben**

```toml
# nemesis-parse/pyproject.toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "nemesis-parse"
version = "0.1.0"
description = "Tree-sitter AST parser for Nemesis"
requires-python = ">=3.11"

[tool.maturin]
features = ["pyo3/extension-module"]
```

**Step 3: Minimales lib.rs schreiben**

```rust
// nemesis-parse/src/lib.rs
use pyo3::prelude::*;

/// Returns the version of nemesis-parse.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// nemesis-parse — Tree-sitter AST parser for Nemesis.
#[pymodule]
fn nemesis_parse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

**Step 4: Validierung — Cargo check**
Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo check 2>&1 | tail -5`
Expected: `Finished` (oder `Compiling` gefolgt von `Finished` ohne Errors)

Hinweis: Beim ersten Mal werden Crates heruntergeladen, das kann 1-2 Minuten dauern.

**Step 5: Commit**
```bash
git add nemesis-parse/
git commit -m "feat: add nemesis-parse Rust crate with PyO3 scaffolding"
```

---

### Task 8: GitHub Actions CI (.github/workflows/ci.yml)

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: CI-Workflow schreiben**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  PYTHON_VERSION: "3.11"
  RUST_VERSION: "stable"

jobs:
  lint:
    name: Lint (Ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install ruff
        run: pip install ruff>=0.4
      - name: Ruff check
        run: ruff check nemesis/ tests/
      - name: Ruff format check
        run: ruff format --check nemesis/ tests/

  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install pydantic pydantic-settings click pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: |
          python -m pytest tests/ -v --tb=short --cov=nemesis --cov-report=term-missing
      - name: Upload coverage
        if: matrix.python-version == '3.11'
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: .coverage

  rust-check:
    name: Rust Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: "nemesis-parse -> target"
      - name: Cargo check
        working-directory: nemesis-parse
        run: cargo check
      - name: Cargo test
        working-directory: nemesis-parse
        run: cargo test
      - name: Cargo clippy
        working-directory: nemesis-parse
        run: cargo clippy -- -D warnings

  build-wheel:
    name: Build Wheel (maturin)
    runs-on: ubuntu-latest
    needs: [rust-check]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - uses: dtolnay/rust-toolchain@stable
      - name: Install maturin
        run: pip install maturin>=1.5
      - name: Build wheel
        working-directory: nemesis-parse
        run: maturin build --release
      - name: Upload wheel
        uses: actions/upload-artifact@v4
        with:
          name: wheel
          path: nemesis-parse/target/wheels/*.whl
```

**Step 2: Validierung — YAML Syntax**
Run: `cd /home/andreas/projects/nemesis && python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML valid')"`
Expected: `YAML valid`

Hinweis: Falls `pyyaml` nicht installiert ist:
Run: `cd /home/andreas/projects/nemesis && python3 -c "import json, re; content=open('.github/workflows/ci.yml').read(); print('name:' in content and 'jobs:' in content and 'lint:' in content and 'test:' in content and 'rust-check:' in content)"`
Expected: `True`

**Step 3: Commit**
```bash
git add .github/
git commit -m "ci: add GitHub Actions workflow for lint, test, rust check, and wheel build"
```

---

### Task 9: Ruff-Konformität und erster Formatierungslauf

**Files:**
- Modify: alle bestehenden `.py` Dateien (falls nötig)

**Step 1: Ruff Check ausführen**
Run: `cd /home/andreas/projects/nemesis && python3 -m ruff check nemesis/ tests/`
Expected: `All checks passed!` oder spezifische Fehler

**Step 2: Falls Fehler — Ruff Fix**
Run: `cd /home/andreas/projects/nemesis && python3 -m ruff check --fix nemesis/ tests/`

**Step 3: Ruff Format ausführen**
Run: `cd /home/andreas/projects/nemesis && python3 -m ruff format nemesis/ tests/`

**Step 4: Validierung — alle Tests laufen noch**
Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/ -v`
Expected: PASS (alle Tests bestehen)

**Step 5: Commit**
```bash
git add -A
git commit -m "style: apply ruff formatting and lint fixes"
```

---

## Zusammenfassung

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 6 | Rust Crate Scaffolding | `nemesis-parse/{Cargo.toml,pyproject.toml,src/lib.rs}` | cargo check |
| 8 | GitHub Actions CI | `.github/workflows/ci.yml` | YAML-Validierung |
| 9 | Ruff-Konformität | Bestehende Dateien | Lint + Format |

**Gesamt: 3 Tasks, 0 Python-Tests (Validierung via cargo + ruff), 3 Commits**

---

**Vorheriges Paket:** [01b — Core Infrastructure](01b-core-infrastructure.md)
**Nächstes Paket:** —
