"""Graph module — abstract adapter, Kuzu (default), Neo4j (optional)."""

from __future__ import annotations

from typing import Any

from nemesis.graph.adapter import (
    EDGE_TYPES,
    NODE_SCHEMAS,
    NODE_TYPES,
    EdgeData,
    GraphAdapter,
    NodeData,
    TraversalResult,
)


def create_graph_adapter(
    backend: str = "kuzu",
    create_schema: bool = False,
    **kwargs: Any,
) -> GraphAdapter:
    if backend == "kuzu":
        from nemesis.graph.kuzu import KuzuAdapter

        adapter = KuzuAdapter(**kwargs)
    elif backend == "neo4j":
        from nemesis.graph.neo4j import Neo4jAdapter

        adapter = Neo4jAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown graph backend: {backend!r}. Supported: 'kuzu', 'neo4j'")

    if create_schema:
        adapter.create_schema()
    return adapter


__all__ = [
    "EDGE_TYPES",
    "NODE_SCHEMAS",
    "NODE_TYPES",
    "EdgeData",
    "GraphAdapter",
    "NodeData",
    "TraversalResult",
    "create_graph_adapter",
]
