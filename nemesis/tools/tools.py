"""MCP Tool implementations for Nemesis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemesis.core.engine import NemesisEngine


def search_code(engine: NemesisEngine, query: str, limit: int = 10) -> dict[str, Any]:
    """Search code using vector similarity + optional graph context.

    1. Embed the query text
    2. Search vector store for similar chunks
    3. For each result, optionally load the parent node from graph
    4. Return results with file, code, score, and node info
    """
    embedding = engine.embedder.embed_single(query)
    results = engine.vector_store.search(embedding, limit=limit)

    items = []
    for r in results:
        item: dict[str, Any] = {
            "id": r.id,
            "text": r.text,
            "score": r.score,
            "file": r.metadata.get("file", ""),
            "start_line": r.metadata.get("start_line"),
            "end_line": r.metadata.get("end_line"),
        }
        # Try to enrich from graph
        parent_id = r.metadata.get("parent_node_id")
        if parent_id:
            node = engine.graph.get_node(parent_id)
            if node:
                item["node_type"] = node.node_type
                item["node_name"] = node.properties.get("name", "")
        items.append(item)

    engine.session.add_query(query)
    engine.session.add_result(query, {"results": items})
    return {"query": query, "results": items, "count": len(items)}


def get_context(
    engine: NemesisEngine, file_path: str, depth: int = 2
) -> dict[str, Any]:
    """Get context for a file from the graph.

    1. Find all nodes for the file
    2. Traverse from each node to get related context
    3. Return structured context with nodes and relationships
    """
    nodes = engine.graph.get_nodes_for_file(file_path)

    context_nodes: list[dict[str, Any]] = []
    related: list[dict[str, Any]] = []

    for node in nodes:
        context_nodes.append({
            "id": node.id,
            "type": node.node_type,
            "name": node.properties.get("name", ""),
            "start_line": node.properties.get("start_line"),
            "end_line": node.properties.get("end_line"),
        })

        # Get neighbors for context
        neighbors = engine.graph.get_neighbors(node.id)
        for neighbor in neighbors:
            if neighbor.id != node.id:
                related.append({
                    "id": neighbor.id,
                    "type": neighbor.node_type,
                    "name": neighbor.properties.get("name", ""),
                })

    # Deduplicate related by id
    seen_ids: set[str] = set()
    unique_related: list[dict[str, Any]] = []
    for r in related:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            unique_related.append(r)

    return {
        "file": file_path,
        "nodes": context_nodes,
        "related": unique_related,
        "node_count": len(context_nodes),
    }


def index_project(
    engine: NemesisEngine,
    path: str,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """Index a project directory.

    Uses the engine's pipeline to index all matching files.
    """
    langs = languages or engine.config.languages
    result = engine.pipeline.index_project(
        Path(path),
        languages=langs,
        ignore_dirs=set(engine.config.ignore_patterns),
    )
    return {
        "files_indexed": result.files_indexed,
        "nodes_created": result.nodes_created,
        "edges_created": result.edges_created,
        "chunks_created": result.chunks_created,
        "embeddings_created": result.embeddings_created,
        "duration_ms": round(result.duration_ms, 1),
        "errors": result.errors,
        "success": result.success,
    }


def update_project(
    engine: NemesisEngine,
    path: str,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """Run a delta update on a project directory."""
    langs = languages or engine.config.languages
    result = engine.pipeline.update_project(
        Path(path),
        languages=langs,
        ignore_dirs=set(engine.config.ignore_patterns),
    )
    return {
        "files_indexed": result.files_indexed,
        "nodes_created": result.nodes_created,
        "edges_created": result.edges_created,
        "duration_ms": round(result.duration_ms, 1),
        "errors": result.errors,
        "success": result.success,
    }


def remember_rule(
    engine: NemesisEngine,
    content: str,
    scope: str = "project",
    source: str = "user",
) -> dict[str, Any]:
    """Remember a coding rule."""
    rule = engine.rules.add_rule(content, scope=scope, source=source)
    return {
        "id": rule.id,
        "content": rule.content,
        "scope": rule.scope,
        "source": rule.source,
        "created_at": str(rule.created_at),
    }


def remember_decision(
    engine: NemesisEngine,
    title: str,
    reasoning: str = "",
    status: str = "accepted",
) -> dict[str, Any]:
    """Remember an architectural decision."""
    decision = engine.decisions.add_decision(title, status=status)
    return {
        "id": decision.id,
        "title": decision.title,
        "status": decision.status,
        "created_at": str(decision.created_at),
    }


def get_memory(engine: NemesisEngine) -> dict[str, Any]:
    """Get all stored memory (rules, decisions, conventions)."""
    rules = engine.rules.get_rules()
    decisions = engine.decisions.get_decisions()
    conventions = engine.conventions.get_conventions()

    return {
        "rules": [
            {"id": r.id, "content": r.content, "scope": r.scope} for r in rules
        ],
        "decisions": [
            {"id": d.id, "title": d.title, "status": d.status} for d in decisions
        ],
        "conventions": [
            {"id": c.id, "pattern": c.pattern, "scope": c.scope} for c in conventions
        ],
        "total": len(rules) + len(decisions) + len(conventions),
    }


def get_session_summary(engine: NemesisEngine) -> dict[str, Any]:
    """Get a summary of the current session context."""
    return {
        "queries": engine.session.get_queries(),
        "results": engine.session.get_results(),
        "summary": engine.session.build_summary(),
    }
