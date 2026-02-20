# Nemesis

**GraphRAG context engine for AI coding agents.**

Nemesis parses your codebase into a semantic knowledge graph, creates vector embeddings for natural-language search, and exposes everything as MCP tools — giving AI agents deep, structured understanding of your code.

```
Your Code  ──>  Rust Parser  ──>  Code Graph (Kuzu)  ──>  MCP Tools  ──>  AI Agent
                                  Vector DB (LanceDB)
                                  Memory (Rules/Decisions)
```

## Features

- **Code Graph** — builds a rich knowledge graph with classes, functions, imports, inheritance, and call relationships. Backed by [Kuzu](https://kuzudb.com/) (embedded) or Neo4j (remote).
- **Semantic Search** — vector similarity search over code chunks using OpenAI embeddings (`text-embedding-3-small`) or local models via sentence-transformers. Powered by [LanceDB](https://lancedb.com/).
- **AST-Aware Chunking** — Rust-based parser (Tree-sitter) splits code at natural boundaries, not arbitrary line counts.
- **Delta Updates** — tracks file hashes and only re-indexes what changed. Auto-compacts the vector store after each update.
- **Memory System** — stores coding rules, architectural decisions, and conventions in the graph for persistent context across sessions.
- **File Watcher** — watches your project for changes and auto-reindexes with configurable debouncing.
- **Multi-Project** — index and query multiple projects independently with isolated namespaces.
- **15+ Languages** — Python, TypeScript, JavaScript, Rust, Go, Java, C/C++, C#, Ruby, PHP, Swift, Kotlin, Scala, Zig.

## Quick Start

### Installation

```bash
pip install nemesis-ai
```

With optional backends:

```bash
pip install nemesis-ai[neo4j]             # Neo4j graph backend
pip install nemesis-ai[local-embeddings]  # Local embeddings (no API key needed)
```

### Configuration

Set your OpenAI API key (or use `local` provider):

```bash
echo "NEMESIS_OPENAI_API_KEY=sk-..." > ~/.nemesis/.env
```

All settings via environment variables with `NEMESIS_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEMESIS_DATA_DIR` | `~/.nemesis` | Data storage directory |
| `NEMESIS_GRAPH_BACKEND` | `kuzu` | Graph backend (`kuzu` or `neo4j`) |
| `NEMESIS_VECTOR_PROVIDER` | `openai` | Embedding provider (`openai` or `local`) |
| `NEMESIS_VECTOR_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `NEMESIS_LANGUAGES` | `["python"]` | Languages to index |

### CLI

```bash
nemesis index ./my-project              # Index a project
nemesis query "authentication logic"    # Semantic search
nemesis watch ./my-project              # Watch & auto-reindex
nemesis projects                        # List indexed projects
nemesis serve                           # Start MCP server
```

### MCP Server Setup

Add to your Claude Code config (`~/.claude.json`):

```json
{
  "mcpServers": {
    "nemesis": {
      "command": "/path/to/nemesis",
      "args": ["serve"],
      "env": {
        "NEMESIS_OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_code` | Natural-language semantic search over your codebase |
| `get_context` | Get full graph context for a file (classes, methods, imports, relations) |
| `index_project` | Index a project directory |
| `update_project` | Delta update — only re-indexes changed files |
| `remember_rule` | Store a coding rule for persistent context |
| `remember_decision` | Record an architectural decision |
| `get_memory` | Retrieve all stored rules, decisions, and conventions |
| `get_session_summary` | Get current session queries and results |
| `list_projects` | List all indexed projects |
| `remove_project` | Remove a project and its indexed data |

## Architecture

```
nemesis/
├── core/           # Engine, CLI, MCP server, config, file watcher
├── graph/          # Graph DB abstraction (Kuzu + Neo4j adapters)
├── vector/         # LanceDB vector store + embedding providers
├── indexer/        # Parsing pipeline, delta detection, AST chunking
├── parser/         # Rust parser bridge (PyO3 + Tree-sitter)
├── memory/         # Rules, decisions, conventions, session context
└── tools/          # MCP tool implementations
```

**Graph Schema**: 15 node types, 16 edge types covering code structure (File, Class, Function, Method, Variable, Import), relationships (CONTAINS, INHERITS, CALLS, IMPORTS), and knowledge (Rule, Decision, Convention).

**Data Storage** (`~/.nemesis/`):
- `graph/` — Kuzu embedded database
- `vectors/` — LanceDB vector store
- `registry.json` — Project metadata

## Tech Stack

| Component | Technology |
|-----------|------------|
| Code Parsing | Rust + Tree-sitter (via PyO3) |
| Graph DB | Kuzu (embedded) / Neo4j (remote) |
| Vector DB | LanceDB |
| Embeddings | OpenAI `text-embedding-3-small` / sentence-transformers |
| Search Metric | Cosine similarity |
| Protocol | MCP (Model Context Protocol) |
| Config | Pydantic Settings |
| CLI | Click |

## Development

```bash
git clone https://github.com/CytrexSGR/Nemesis.git
cd Nemesis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

487 tests, Python 3.11+.

## License

[MIT](LICENSE)
