"""Abstract graph adapter protocol for Nemesis.

Defines the common interface that all graph backends (Kuzu, Neo4j)
must implement. Uses Python's Protocol for structural subtyping --
no inheritance required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Schema constants -- all node and edge types from the design document
# ---------------------------------------------------------------------------

NODE_TYPES: set[str] = {
    # Code nodes (from Tree-sitter)
    "File",
    "Module",
    "Class",
    "Function",
    "Method",
    "Interface",
    "Variable",
    "Import",
    # Memory nodes (from developer)
    "Rule",
    "Decision",
    "Alternative",
    "Convention",
    # Meta nodes
    "Project",
    "Chunk",
}

EDGE_TYPES: set[str] = {
    # Code edges
    "CONTAINS",
    "HAS_METHOD",
    "INHERITS",
    "IMPLEMENTS",
    "CALLS",
    "IMPORTS",
    "RETURNS",
    "ACCEPTS",
    # Memory edges
    "APPLIES_TO",
    "CHOSE",
    "REJECTED",
    "GOVERNS",
    # Chunk edges
    "CHUNK_OF",
    # Meta edges
    "HAS_FILE",
}

# ---------------------------------------------------------------------------
# Node property schemas -- required and optional properties per node type
# ---------------------------------------------------------------------------

NODE_SCHEMAS: dict[str, dict[str, str]] = {
    "File": {
        "id": "STRING",
        "path": "STRING",
        "language": "STRING",
        "hash": "STRING",
        "last_indexed": "STRING",
        "size": "INT64",
    },
    "Module": {
        "id": "STRING",
        "name": "STRING",
        "path": "STRING",
        "docstring": "STRING",
    },
    "Class": {
        "id": "STRING",
        "name": "STRING",
        "file": "STRING",
        "line_start": "INT64",
        "line_end": "INT64",
        "docstring": "STRING",
    },
    "Function": {
        "id": "STRING",
        "name": "STRING",
        "file": "STRING",
        "line_start": "INT64",
        "line_end": "INT64",
        "signature": "STRING",
        "docstring": "STRING",
        "is_async": "BOOL",
    },
    "Method": {
        "id": "STRING",
        "name": "STRING",
        "class_name": "STRING",
        "file": "STRING",
        "line_start": "INT64",
        "line_end": "INT64",
        "signature": "STRING",
        "visibility": "STRING",
    },
    "Interface": {
        "id": "STRING",
        "name": "STRING",
        "file": "STRING",
        "language": "STRING",
    },
    "Variable": {
        "id": "STRING",
        "name": "STRING",
        "file": "STRING",
        "type_hint": "STRING",
        "scope": "STRING",
    },
    "Import": {
        "id": "STRING",
        "name": "STRING",
        "source": "STRING",
        "alias": "STRING",
    },
    "Rule": {
        "id": "STRING",
        "content": "STRING",
        "scope": "STRING",
        "created_at": "STRING",
        "source": "STRING",
    },
    "Decision": {
        "id": "STRING",
        "title": "STRING",
        "reasoning": "STRING",
        "created_at": "STRING",
        "status": "STRING",
    },
    "Alternative": {
        "id": "STRING",
        "title": "STRING",
        "reason_rejected": "STRING",
    },
    "Convention": {
        "id": "STRING",
        "pattern": "STRING",
        "example": "STRING",
        "scope": "STRING",
    },
    "Project": {
        "id": "STRING",
        "name": "STRING",
        "root_path": "STRING",
        "languages": "STRING",
        "last_indexed": "STRING",
    },
    "Chunk": {
        "id": "STRING",
        "content": "STRING",
        "token_count": "INT64",
        "embedding_id": "STRING",
        "parent_type": "STRING",
    },
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeData:
    """A node to be stored in the graph.

    Attributes:
        id: Unique identifier for the node.
        node_type: One of NODE_TYPES (e.g. "Function", "Class", "File").
        properties: Arbitrary key-value properties for the node.
    """

    id: str
    node_type: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeData:
    """An edge (relationship) to be stored in the graph.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        edge_type: One of EDGE_TYPES (e.g. "CALLS", "CONTAINS").
        properties: Arbitrary key-value properties for the edge.
    """

    source_id: str
    target_id: str
    edge_type: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TraversalResult:
    """Result of a graph traversal operation.

    Contains all nodes and edges found during the traversal,
    starting from a given node and following specified edge types
    up to a maximum depth.

    Attributes:
        nodes: All nodes discovered during traversal.
        edges: All edges traversed.
    """

    nodes: list[NodeData]
    edges: list[EdgeData]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphAdapter(Protocol):
    """Protocol for graph database backends.

    Any class implementing these methods can serve as a graph store.
    Two concrete implementations ship with Nemesis:
    KuzuAdapter (embedded, default) and Neo4jAdapter (optional).
    """

    def create_schema(self) -> None:
        """Create all node and edge tables in the database.

        Must be idempotent -- calling it on an existing schema
        should not raise or destroy data.
        """
        ...

    def add_node(self, node: NodeData) -> None:
        """Add or update a node in the graph.

        If a node with the same ID already exists, it is updated.

        Args:
            node: The node data to add.
        """
        ...

    def add_edge(self, edge: EdgeData) -> None:
        """Add an edge between two existing nodes.

        Args:
            edge: The edge data to add.
        """
        ...

    def get_node(self, node_id: str) -> NodeData | None:
        """Retrieve a node by its ID.

        Args:
            node_id: The unique node identifier.

        Returns:
            NodeData if found, None otherwise.
        """
        ...

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[NodeData]:
        """Get neighboring nodes connected by edges.

        Args:
            node_id: The starting node ID.
            edge_type: Optional filter for edge type.
            direction: "outgoing", "incoming", or "both".

        Returns:
            List of neighboring NodeData.
        """
        ...

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> TraversalResult:
        """Traverse the graph from a starting node.

        Performs a breadth-first traversal, following specified
        edge types up to max_depth hops.

        Args:
            start_id: The starting node ID.
            edge_types: Optional list of edge types to follow.
            max_depth: Maximum traversal depth.

        Returns:
            TraversalResult with all discovered nodes and edges.
        """
        ...

    def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Execute a raw Cypher query.

        Args:
            cypher: The Cypher query string.
            parameters: Optional query parameters.

        Returns:
            List of result rows as dicts.
        """
        ...

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges.

        Args:
            node_id: The node ID to delete.
        """
        ...

    def delete_edges_for_file(self, file_path: str) -> None:
        """Delete all edges associated with nodes from a specific file.

        Args:
            file_path: The file path whose edges should be removed.
        """
        ...

    def get_file_hashes(self) -> dict[str, str]:
        """Get all stored file hashes.

        Returns:
            Dict mapping file path to SHA-256 hash.
        """
        ...

    def get_nodes_for_file(self, file_path: str) -> list[NodeData]:
        """Get all nodes associated with a file.

        Args:
            file_path: The file path to query.

        Returns:
            List of NodeData belonging to the file.
        """
        ...

    def get_chunk_ids_for_file(self, file_path: str) -> list[str]:
        """Get IDs of all Chunk nodes linked to a file's nodes.

        Args:
            file_path: The file path to query.

        Returns:
            List of chunk node IDs.
        """
        ...

    def delete_nodes_for_file(self, file_path: str) -> None:
        """Delete all nodes and edges associated with a file.

        Used during delta updates to remove stale data before
        re-indexing a changed file.

        Args:
            file_path: The file path whose nodes should be removed.
        """
        ...

    def clear(self) -> None:
        """Delete all nodes and edges from the graph.

        Used for testing and full re-index.
        """
        ...

    def close(self) -> None:
        """Close the database connection and release resources."""
        ...
