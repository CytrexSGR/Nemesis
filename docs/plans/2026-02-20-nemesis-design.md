# Nemesis — Design Document

> Graph-powered context engine for AI coding agents.
> Combines persistent memory, code intelligence, and dynamic context loading via MCP.

**Date:** 2026-02-20
**Status:** Approved
**Repository:** github.com/CytrexSGR/nemesis

---

## Problem

AI coding assistants (Claude Code, Cursor, Copilot) rely on static markdown files (CLAUDE.md, ARCHITECTURE.md, .cursorrules) to understand a project. This approach has fundamental scaling problems:

- **Context bloat:** A real-world project can accumulate 250+ KB of markdown that gets loaded every session, consuming a quarter of the context window before the user asks a single question.
- **No selectivity:** Everything is loaded regardless of the current task. Working on the frontend? The entire backend architecture doc is still in context.
- **Stale data:** Markdown files drift out of sync with the actual codebase. Architecture docs describe code that no longer exists.
- **No structure:** Plain text can't express relationships. "ServiceA calls ServiceB which depends on LibC" is prose, not queryable data.

Classical RAG (vector search over code chunks) partially solves this but fails at understanding code structure. It finds the file `login.ts` but doesn't know that `login.ts` calls `auth_utils.ts` which implements an interface from `types.d.ts`. The context is broken, the AI hallucinates.

## Solution

Nemesis is a local GraphRAG system for code, packaged as an MCP server. It combines:

1. **Code Intelligence** — Tree-sitter AST parsing builds a knowledge graph of code structure (classes, functions, calls, imports, inheritance).
2. **Semantic Search** — Vector embeddings enable natural language queries over code.
3. **Persistent Memory** — Developer rules, architecture decisions, and conventions are stored as graph nodes and automatically included in relevant context.
4. **Dynamic Context Loading** — Instead of loading everything, Nemesis delivers a curated context package sized to fit the token budget, containing only what's relevant to the current task.

---

## Architecture

### Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Core** | Python 3.11+ | AI/ML ecosystem, MCP SDK, fast distribution via uv/pip |
| **Parser** | Rust (PyO3/maturin) | Tree-sitter performance for AST extraction |
| **Graph DB** | Kuzu (embedded, default) / Neo4j (optional) | Kuzu: zero-config local. Neo4j: existing infrastructure |
| **Vector DB** | LanceDB (embedded) | Local-first, no server needed |
| **Embeddings** | OpenAI text-embedding-3-small (default) | Best quality. Local option via sentence-transformers |
| **MCP** | MCP SDK (Python) | Claude Code integration, LSP-ready architecture for later |
| **CLI** | Click | User-facing commands |
| **Config** | Pydantic | Validation, type safety |

### Design Principle: Pydantic/Ruff Pattern

CPU-intensive work (AST parsing of large codebases) is written in Rust and compiled as a Python extension via maturin/PyO3. Everything else (MCP server, AI layer, DB adapters, memory) stays in Python. Users install via `pip install nemesis-ai` with pre-compiled wheels — no Rust compiler needed.

### Directory Structure

```
nemesis/
├── nemesis/                     # Python package
│   ├── core/
│   │   ├── server.py            # MCP server (stdio transport)
│   │   ├── config.py            # Project config, DB selection
│   │   ├── cli.py               # CLI: nemesis index, query, watch, serve
│   │   └── watcher.py           # File watcher (watchdog) -> delta updates
│   ├── indexer/
│   │   ├── pipeline.py          # Orchestrates: Parse -> Chunk -> Embed -> Store
│   │   ├── chunker.py           # AST-aware chunking (large nodes -> sub-chunks)
│   │   └── delta.py             # Diff logic: delete old nodes, insert new ones
│   ├── parser/
│   │   └── bridge.py            # PyO3 interface to nemesis-parse
│   ├── graph/
│   │   ├── adapter.py           # Common interface (abstract)
│   │   ├── neo4j.py             # Neo4j driver adapter
│   │   └── kuzu.py              # Kuzu embedded adapter
│   ├── vector/
│   │   ├── embeddings.py        # OpenAI (default) + local option
│   │   └── store.py             # Vector index (LanceDB)
│   ├── memory/
│   │   ├── rules.py             # Project rules
│   │   ├── decisions.py         # Architecture decisions
│   │   ├── context.py           # Session context, history
│   │   └── auto_learn.py        # Pattern detection in conversations
│   └── tools/
│       ├── code_query.py        # get_code_semantics(), get_dependencies()
│       ├── memory_query.py      # get_project_context(), store_rule()
│       └── index_tools.py       # index_project(), watch_project()
├── nemesis-parse/               # Rust crate (maturin/PyO3)
│   ├── src/
│   │   ├── lib.rs               # PyO3 bindings
│   │   ├── parser.rs            # Tree-sitter multi-language AST
│   │   ├── extractor.rs         # Nodes + edges from AST
│   │   └── languages/           # Grammar bindings
│   └── Cargo.toml
├── pyproject.toml
├── tests/
└── docs/
```

---

## Data Flow

### Full Index

```
Code files on disk
  -> nemesis-parse (Rust/Tree-sitter) -> AST -> Nodes + Edges
  -> chunker.py: Large nodes -> logical sub-chunks
  -> embeddings.py (OpenAI API) -> Vector embeddings
  -> graph adapter: Store nodes + edges in Kuzu/Neo4j
  -> vector store: Store embeddings in LanceDB
```

### Delta Update (File Changed)

```
File changed (watchdog event)
  -> delta.py: Compare hash, find old nodes for this file in graph
  -> nemesis-parse: Parse only this single file
  -> chunker.py: Split large AST nodes into sub-chunks
  -> vector store: Delete old embeddings, create new ones
  -> graph adapter: Delete old nodes/edges, insert new ones
  Result: Incremental update in milliseconds, not minutes
```

### Query

```
Query via MCP tool (e.g. "How does the auth flow work?")
  -> Semantic search (vector) -> Find entry points
  -> Graph traversal -> Expand dependencies, callers, architecture
  -> Memory layer -> Mix in relevant rules and decisions
  -> Context ranking -> Trim to token budget
  -> Return curated context package to Claude
```

---

## Graph Schema

### Node Types

```
CODE NODES (from Tree-sitter):
  :File          {path, language, hash, last_indexed, size}
  :Module        {name, path, docstring}
  :Class         {name, file, line_start, line_end, docstring}
  :Function      {name, file, line_start, line_end, signature, docstring, is_async}
  :Method        {name, class, file, line_start, line_end, signature, visibility}
  :Interface     {name, file, language}
  :Variable      {name, file, type_hint, scope}
  :Import        {name, source, alias}

MEMORY NODES (from developer):
  :Rule          {content, scope, created_at, source}
  :Decision      {title, reasoning, created_at, status}
  :Alternative   {title, reason_rejected}
  :Convention    {pattern, example, scope}

META NODES:
  :Project       {name, root_path, languages[], last_indexed}
  :Chunk         {content, token_count, embedding_id, parent_type}
```

### Edge Types

```
CODE EDGES:
  (:File)-[:CONTAINS]->(:Class|:Function|:Variable)
  (:Class)-[:HAS_METHOD]->(:Method)
  (:Class)-[:INHERITS]->(:Class)
  (:Class)-[:IMPLEMENTS]->(:Interface)
  (:Function)-[:CALLS]->(:Function|:Method)
  (:Function)-[:IMPORTS]->(:Module|:File)
  (:Function)-[:RETURNS]->(:Class)
  (:Function)-[:ACCEPTS]->(:Class)
  (:File)-[:IMPORTS]->(:File)

MEMORY EDGES:
  (:Rule)-[:APPLIES_TO]->(:File|:Module|:Class|:Project)
  (:Decision)-[:CHOSE]->(:Convention|:Class|:Module)
  (:Decision)-[:REJECTED]->(:Alternative)
  (:Convention)-[:GOVERNS]->(:Module|:File)

CHUNK EDGES:
  (:Chunk)-[:CHUNK_OF]->(:Class|:Function|:Method|:File)

META EDGES:
  (:Project)-[:HAS_FILE]->(:File)
```

---

## MCP Tools

### Code Intelligence

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `get_code_semantics` | Natural language code search | query, limit | file, function, snippet, related_files, relevance |
| `get_dependencies` | Dependency traversal | symbol, depth, direction | node, deps_in, deps_out, call_chain |
| `get_architecture` | Architecture overview | scope (project/module/file) | modules, key_classes, data_flow, entry_points |
| `get_impact` | Change impact analysis | file, function | direct_dependents, transitive_dependents, test_coverage |

### Memory

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `get_project_context` | Retrieve rules + decisions | topic (optional) | rules, decisions, conventions |
| `store_rule` | Save a project rule | rule, scope, related_to | id, stored |
| `store_decision` | Document architecture decision | title, reasoning, alternatives | id, stored |

### Index Management

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `index_project` | Full project index | path, languages | files_indexed, nodes, edges, duration |
| `index_status` | Check index health | — | last_indexed, files, nodes, edges, stale_files |
| `watch_project` | Start/stop file watcher | path, enabled | watching, path, pid |

### Smart Context (Core Tool)

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `get_smart_context` | CLAUDE.md replacement | task, max_tokens | code_context, rules, decisions, architecture, token_count |

`get_smart_context` is the central tool that replaces static markdown files. It combines code intelligence and memory into a single curated context package, ranked by relevance and trimmed to the token budget.

---

## Hooks & Automation

### Claude Code Integration

```json
{
  "hooks": {
    "SessionStart": [{
      "command": "nemesis hook session-start --project $PWD",
      "timeout": 5000
    }],
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "command": "nemesis hook file-changed --file $FILE",
      "timeout": 3000
    }],
    "PreToolUse": [{
      "matcher": "Task|EnterPlanMode",
      "command": "nemesis hook pre-task --project $PWD",
      "timeout": 3000
    }]
  }
}
```

### Hook Behavior

**SessionStart** — The CLAUDE.md killer:
1. Check if project is indexed (prompt to index if not)
2. Run delta updates for stale files in background
3. Load base context: active rules, recent decisions, architecture overview
4. Output compact context package (< 2000 tokens)

**PostToolUse (Edit/Write)** — Incremental index:
1. Identify changed file
2. Delete old AST nodes for this file from graph
3. Re-parse, re-chunk, re-embed
4. Update graph silently in background

**PreToolUse (Task/Plan)** — Context enrichment:
1. Read the task/plan prompt
2. Call `get_smart_context(task=prompt, max_tokens=4000)`
3. Inject relevant code + rules + architecture into context

### Auto-Learning

Nemesis detects patterns in conversations and automatically creates memory nodes:

- "ab jetzt immer..." / "always use..." -> Rule
- "wir haben entschieden..." / "we decided..." -> Decision
- "nutze nie..." / "never use..." -> Rule (negative)
- "Convention:" -> Convention

---

## Packaging & Distribution

### Installation

```bash
# Recommended
uv tool install nemesis-ai

# Classic
pip install nemesis-ai

# With optional features
pip install nemesis-ai[neo4j]              # Neo4j support
pip install nemesis-ai[local-embeddings]   # No cloud API needed
```

### Pre-compiled Wheels

Rust extension `nemesis-parse` ships as pre-compiled wheels via maturin:
- Linux x86_64 + aarch64
- macOS x86_64 + ARM (Apple Silicon)
- Windows x86_64

No Rust compiler needed on user machines.

### Project Config

```toml
# .nemesis/config.toml (in project root)
[project]
name = "my-project"
languages = ["python", "typescript"]
ignore = ["node_modules", "venv", "__pycache__", ".git"]

[graph]
backend = "kuzu"  # or "neo4j"

[vector]
provider = "openai"
model = "text-embedding-3-small"

[memory]
auto_load_rules = true
auto_learn = true

[watcher]
enabled = true
debounce_ms = 500
```

---

## Typical Session Flow

```
1. Developer opens Claude Code in ~/eve_copilot/
2. Hook: SessionStart
   -> Nemesis delivers: "Python/FastAPI project, 16 microservices,
      3 active rules, last decision: SharedLib refactoring"
3. Developer: "Refactor the market service"
4. Hook: PreToolUse (Plan)
   -> Nemesis delivers: MarketService classes, dependencies,
      rule "parameterized queries", architecture context
5. Claude plans with full context without loading 250 KB of markdown
6. Claude edits market_service.py
7. Hook: PostToolUse (Edit)
   -> Nemesis updates graph in background
8. Next query immediately has current state
```

---

## Compatibility

- **Primary:** Claude Code (MCP)
- **Planned:** LSP for IDE integration (Cursor, VS Code, Windsurf)
- **Architecture:** MCP-first, LSP-ready (core logic decoupled from transport)

## Success Criteria

1. Starting a Claude Code session on the EVE Copilot project loads relevant context from the graph instead of 250 KB markdown
2. Asking "How does the market service work?" returns relevant files, dependencies, and architecture decisions
3. Index updates happen incrementally on file save, not requiring full re-index
