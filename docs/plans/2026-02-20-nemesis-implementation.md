# Nemesis Implementation Master Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Nemesis — a GraphRAG context engine that replaces static CLAUDE.md files with intelligent, graph-based dynamic context loading via MCP.

**Architecture:** Python 3.11+ core with Rust extension (PyO3/maturin) for high-performance AST parsing via Tree-sitter. Embedded databases (Kuzu for graph, LanceDB for vectors) ensure zero-config local operation. MCP server provides Claude Code integration. Memory system stores rules, decisions, and conventions as graph nodes.

**Tech Stack:** Python 3.11+, Rust/PyO3/maturin, Tree-sitter, Kuzu, LanceDB, OpenAI Embeddings, MCP SDK, Click, Pydantic, watchdog

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

---

## Implementation Phases

### Phase 1: Foundation

| Plan | Component | Dependencies | Estimated Tasks |
|------|-----------|-------------|-----------------|
| [01-project-setup](01-project-setup.md) | Project scaffolding, pyproject.toml, Cargo.toml, CI, dev tooling | None | ~8 |

### Phase 2: Core Components (parallel)

| Plan | Component | Dependencies | Estimated Tasks |
|------|-----------|-------------|-----------------|
| [02-rust-parser](02-rust-parser.md) | nemesis-parse Rust crate, Tree-sitter, PyO3 bindings | 01 | ~12 |
| [03-graph-layer](03-graph-layer.md) | Abstract graph adapter, Kuzu implementation, schema | 01 | ~10 |
| [04-vector-store](04-vector-store.md) | LanceDB integration, embedding generation, search | 01 | ~8 |

### Phase 3: Integration

| Plan | Component | Dependencies | Estimated Tasks |
|------|-----------|-------------|-----------------|
| [05-indexing-pipeline](05-indexing-pipeline.md) | Parse → Chunk → Embed → Store orchestration, delta updates | 02, 03, 04 | ~10 |

### Phase 4: Interface

| Plan | Component | Dependencies | Estimated Tasks |
|------|-----------|-------------|-----------------|
| [06-mcp-server](06-mcp-server.md) | MCP server (stdio), all code intelligence & memory tools | 05 | ~12 |

### Phase 5: Intelligence (parallel)

| Plan | Component | Dependencies | Estimated Tasks |
|------|-----------|-------------|-----------------|
| [07-memory-system](07-memory-system.md) | Rules, decisions, conventions, auto-learning | 03, 06 | ~10 |
| [08-cli-hooks](08-cli-hooks.md) | Click CLI, Claude Code hooks, file watcher | 06 | ~10 |

### Phase 6: Final Integration

- End-to-end tests across all components
- Packaging & distribution (maturin wheels)
- README & documentation

---

## Dependency Graph

```
Phase 1        Phase 2 (parallel)       Phase 3          Phase 4         Phase 5 (parallel)
─────────────────────────────────────────────────────────────────────────────────────────────

01-project  ──┬── 02-rust-parser  ──┐
              ├── 03-graph-layer  ──┼── 05-indexing  ── 06-mcp-server ─┬── 07-memory-system
              └── 04-vector-store ──┘                                  └── 08-cli-hooks
```

## Repository Structure

```
nemesis/
├── nemesis/                     # Python package
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── server.py            # MCP server (stdio transport)
│   │   ├── config.py            # Pydantic config models
│   │   ├── cli.py               # Click CLI
│   │   └── watcher.py           # File watcher (watchdog)
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Orchestrates: Parse → Chunk → Embed → Store
│   │   ├── chunker.py           # AST-aware chunking
│   │   └── delta.py             # Diff logic for incremental updates
│   ├── parser/
│   │   ├── __init__.py
│   │   └── bridge.py            # PyO3 interface to nemesis-parse
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── adapter.py           # Abstract graph protocol
│   │   ├── kuzu.py              # Kuzu embedded adapter
│   │   └── neo4j.py             # Neo4j adapter (optional)
│   ├── vector/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # OpenAI (default) + local option
│   │   └── store.py             # LanceDB vector index
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── rules.py             # Project rules
│   │   ├── decisions.py         # Architecture decisions
│   │   ├── context.py           # Session context
│   │   └── auto_learn.py        # Pattern detection
│   └── tools/
│       ├── __init__.py
│       ├── code_query.py        # get_code_semantics, get_dependencies, etc.
│       ├── memory_query.py      # get_project_context, store_rule, etc.
│       └── index_tools.py       # index_project, watch_project, etc.
├── nemesis-parse/               # Rust crate (maturin/PyO3)
│   ├── src/
│   │   ├── lib.rs               # PyO3 module bindings
│   │   ├── parser.rs            # Tree-sitter multi-language AST
│   │   ├── extractor.rs         # Extract nodes + edges from AST
│   │   └── languages/           # Grammar bindings
│   ├── Cargo.toml
│   └── pyproject.toml           # maturin build config
├── tests/
│   ├── conftest.py
│   ├── test_parser/
│   ├── test_graph/
│   ├── test_vector/
│   ├── test_indexer/
│   ├── test_server/
│   ├── test_memory/
│   └── test_cli/
├── pyproject.toml
├── README.md
└── docs/
    └── plans/
```

## Execution Strategy

Each sub-plan follows strict TDD:
1. Write failing test
2. Run — verify it fails
3. Implement minimal code
4. Run — verify it passes
5. Commit

Sub-plans in the same phase can be executed in parallel (separate worktrees or agents).

## Ruflo Integration

Ruflo is used throughout development for:
- **Memory:** Store implementation decisions and patterns discovered during development
- **Hooks/Intelligence:** Track trajectory of each phase, learn from successes/failures
- **Workflows:** Orchestrate multi-phase execution
- **Performance:** Benchmark critical paths (parser, indexing, queries)
