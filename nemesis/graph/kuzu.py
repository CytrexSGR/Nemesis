"""Kuzu embedded graph adapter for Nemesis.

Provides a zero-config, embedded graph database using Kuzu.
This is the default backend -- no external services required.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import kuzu

from nemesis.graph.adapter import (
    EDGE_TYPES,
    NODE_SCHEMAS,
    NODE_TYPES,
    EdgeData,
    NodeData,
    TraversalResult,
)

logger = logging.getLogger(__name__)


# Mapping of edge types to valid (source_type, target_type) pairs.
# Kuzu requires REL tables to reference specific node tables.
EDGE_ENDPOINTS: dict[str, list[tuple[str, str]]] = {
    "CONTAINS": [
        ("File", "Class"),
        ("File", "Function"),
        ("File", "Variable"),
        ("File", "Import"),
        ("File", "Interface"),
        ("Module", "Class"),
        ("Module", "Function"),
        ("Module", "Variable"),
    ],
    "HAS_METHOD": [("Class", "Method")],
    "INHERITS": [("Class", "Class")],
    "IMPLEMENTS": [("Class", "Interface")],
    "CALLS": [
        ("Function", "Function"),
        ("Function", "Method"),
        ("Method", "Function"),
        ("Method", "Method"),
    ],
    "IMPORTS": [
        ("Function", "Module"),
        ("Function", "File"),
        ("File", "File"),
        ("File", "Module"),
        ("Module", "Module"),
    ],
    "RETURNS": [("Function", "Class"), ("Method", "Class")],
    "ACCEPTS": [("Function", "Class"), ("Method", "Class")],
    "APPLIES_TO": [
        ("Rule", "File"),
        ("Rule", "Module"),
        ("Rule", "Class"),
        ("Rule", "Project"),
    ],
    "CHOSE": [
        ("Decision", "Convention"),
        ("Decision", "Class"),
        ("Decision", "Module"),
    ],
    "REJECTED": [("Decision", "Alternative")],
    "GOVERNS": [("Convention", "Module"), ("Convention", "File")],
    "CHUNK_OF": [
        ("Chunk", "Class"),
        ("Chunk", "Function"),
        ("Chunk", "Method"),
        ("Chunk", "File"),
    ],
    "HAS_FILE": [("Project", "File")],
}


class KuzuAdapter:
    """Kuzu embedded graph database adapter.

    Args:
        db_path: Directory path for Kuzu database storage.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._schema_created = False

    def create_schema(self) -> None:
        """Create all node and relationship tables.

        Idempotent: skips tables that already exist.
        """
        existing_tables = self._get_existing_tables()

        # Create node tables
        for node_type in sorted(NODE_TYPES):
            if node_type in existing_tables:
                continue
            schema = NODE_SCHEMAS.get(node_type, {"id": "STRING"})
            columns = ", ".join(f"{col} {dtype}" for col, dtype in schema.items() if col != "id")
            ddl = f"CREATE NODE TABLE {node_type}(id STRING, {columns}, PRIMARY KEY (id))"
            try:
                self._conn.execute(ddl)
                logger.debug("Created node table: %s", node_type)
            except RuntimeError as e:
                if "already exists" not in str(e).lower():
                    raise

        # Create relationship tables
        for edge_type in sorted(EDGE_TYPES):
            endpoints = EDGE_ENDPOINTS.get(edge_type, [])
            for src_type, tgt_type in endpoints:
                table_name = f"{edge_type}_{src_type}_{tgt_type}"
                if table_name in existing_tables:
                    continue
                ddl = f"CREATE REL TABLE {table_name}(FROM {src_type} TO {tgt_type})"
                try:
                    self._conn.execute(ddl)
                    logger.debug("Created rel table: %s", table_name)
                except RuntimeError as e:
                    if "already exists" not in str(e).lower():
                        raise

        self._schema_created = True

    def _get_existing_tables(self) -> set[str]:
        """Query Kuzu for existing table names."""
        tables: set[str] = set()
        try:
            result = self._conn.execute("CALL show_tables() RETURN name")
            while result.has_next():
                row = result.get_next()
                tables.add(row[0])
        except RuntimeError:
            pass
        return tables

    def _find_node_type(self, node_id: str) -> str | None:
        """Find which node table contains a given node ID."""
        for node_type in NODE_TYPES:
            try:
                result = self._conn.execute(
                    f"MATCH (n:{node_type}) WHERE n.id = $id RETURN n.id",
                    parameters={"id": node_id},
                )
                if result.has_next():
                    return node_type
            except RuntimeError:
                continue
        return None

    def add_node(self, node: NodeData) -> None:
        """Add or update a node in the graph."""
        schema = NODE_SCHEMAS.get(node.node_type, {"id": "STRING"})
        props: dict[str, Any] = {"id": node.id}
        for col, dtype in schema.items():
            if col == "id":
                continue
            value = node.properties.get(col)
            if value is None:
                if dtype == "STRING":
                    value = ""
                elif dtype == "INT64":
                    value = 0
                elif dtype == "BOOL":
                    value = False
            props[col] = value

        columns = list(props.keys())

        # Build MERGE with SET clause for upsert
        set_clauses = ", ".join(f"n.{col} = ${col}" for col in columns if col != "id")
        if set_clauses:
            merge_query = f"MERGE (n:{node.node_type} {{id: $id}}) SET {set_clauses}"
        else:
            merge_query = f"MERGE (n:{node.node_type} {{id: $id}})"

        self._conn.execute(merge_query, parameters=props)

    def add_edge(self, edge: EdgeData) -> None:
        """Add an edge between two existing nodes."""
        src_type = self._find_node_type(edge.source_id)
        tgt_type = self._find_node_type(edge.target_id)
        if src_type is None or tgt_type is None:
            logger.warning(
                "Cannot add edge %s: source (%s) or target (%s) not found",
                edge.edge_type,
                edge.source_id,
                edge.target_id,
            )
            return

        table_name = f"{edge.edge_type}_{src_type}_{tgt_type}"
        query = (
            f"MATCH (a:{src_type}), (b:{tgt_type}) "
            f"WHERE a.id = $src AND b.id = $tgt "
            f"CREATE (a)-[:{table_name}]->(b)"
        )
        try:
            self._conn.execute(
                query,
                parameters={"src": edge.source_id, "tgt": edge.target_id},
            )
        except RuntimeError as e:
            logger.warning("Failed to add edge %s: %s", table_name, e)

    def get_node(self, node_id: str) -> NodeData | None:
        """Retrieve a node by its ID."""
        for node_type in NODE_TYPES:
            try:
                schema = NODE_SCHEMAS.get(node_type, {"id": "STRING"})
                columns = ", ".join(f"n.{col}" for col in schema)
                result = self._conn.execute(
                    f"MATCH (n:{node_type}) WHERE n.id = $id RETURN {columns}",
                    parameters={"id": node_id},
                )
                if result.has_next():
                    row = result.get_next()
                    col_names = list(schema.keys())
                    properties: dict[str, object] = {}
                    for i, col in enumerate(col_names):
                        if col != "id":
                            properties[col] = row[i]
                    return NodeData(
                        id=node_id,
                        node_type=node_type,
                        properties=properties,
                    )
            except RuntimeError:
                continue
        return None

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[NodeData]:
        """Get neighboring nodes connected by edges."""
        src_type = self._find_node_type(node_id)
        if src_type is None:
            return []

        neighbors: list[NodeData] = []
        seen_ids: set[str] = set()

        edge_types_to_check = [edge_type] if edge_type else list(EDGE_TYPES)

        for et in edge_types_to_check:
            endpoints = EDGE_ENDPOINTS.get(et, [])
            for s_type, t_type in endpoints:
                self._collect_neighbors(
                    node_id,
                    src_type,
                    s_type,
                    t_type,
                    et,
                    direction,
                    neighbors,
                    seen_ids,
                )

        return neighbors

    def _collect_neighbors(
        self,
        node_id: str,
        src_type: str,
        s_type: str,
        t_type: str,
        edge_type: str,
        direction: str,
        neighbors: list[NodeData],
        seen_ids: set[str],
    ) -> None:
        """Collect neighbor nodes for a given edge endpoint pair."""
        table_name = f"{edge_type}_{s_type}_{t_type}"

        # Outgoing: node is the source
        if direction in ("outgoing", "both") and s_type == src_type:
            self._query_neighbors(
                f"MATCH (a:{s_type})-[:{table_name}]->(b:{t_type}) WHERE a.id = $id RETURN b.id",
                node_id,
                neighbors,
                seen_ids,
            )

        # Incoming: node is the target
        if direction in ("incoming", "both") and t_type == src_type:
            self._query_neighbors(
                f"MATCH (a:{s_type})-[:{table_name}]->(b:{t_type}) WHERE b.id = $id RETURN a.id",
                node_id,
                neighbors,
                seen_ids,
            )

    def _query_neighbors(
        self,
        query: str,
        node_id: str,
        neighbors: list[NodeData],
        seen_ids: set[str],
    ) -> None:
        """Execute a neighbor query and collect results."""
        try:
            result = self._conn.execute(query, parameters={"id": node_id})
            while result.has_next():
                row = result.get_next()
                nid = row[0]
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    node = self.get_node(nid)
                    if node:
                        neighbors.append(node)
        except RuntimeError:
            pass

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> TraversalResult:
        """Traverse the graph from a starting node using BFS."""
        all_nodes: dict[str, NodeData] = {}
        all_edges: list[EdgeData] = []
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(start_id, 0)]

        start_node = self.get_node(start_id)
        if start_node:
            all_nodes[start_id] = start_node

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth >= max_depth:
                continue
            visited.add(current_id)

            # Filter by edge types if specified
            if edge_types is not None:
                for et in edge_types:
                    neighbors = self.get_neighbors(current_id, edge_type=et, direction="outgoing")
                    for neighbor in neighbors:
                        if neighbor.id not in all_nodes:
                            all_nodes[neighbor.id] = neighbor
                            queue.append((neighbor.id, depth + 1))
            else:
                neighbors = self.get_neighbors(current_id, direction="outgoing")
                for neighbor in neighbors:
                    if neighbor.id not in all_nodes:
                        all_nodes[neighbor.id] = neighbor
                        queue.append((neighbor.id, depth + 1))

        return TraversalResult(
            nodes=list(all_nodes.values()),
            edges=all_edges,
        )

    def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Execute a raw Cypher query."""
        result = self._conn.execute(cypher, parameters=parameters or {})
        rows: list[dict] = []
        col_names = result.get_column_names()
        while result.has_next():
            row = result.get_next()
            rows.append(dict(zip(col_names, row, strict=False)))
        return rows

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        node_type = self._find_node_type(node_id)
        if node_type is None:
            return
        # Delete all edges first
        self._delete_edges_for_node(node_id, node_type)
        # Then delete the node
        try:
            self._conn.execute(
                f"MATCH (n:{node_type}) WHERE n.id = $id DELETE n",
                parameters={"id": node_id},
            )
        except RuntimeError as e:
            logger.warning("Failed to delete node %s: %s", node_id, e)

    def _delete_edges_for_node(self, node_id: str, node_type: str) -> None:
        """Delete all edges connected to a specific node."""
        for edge_type, endpoints in EDGE_ENDPOINTS.items():
            for s_type, t_type in endpoints:
                table_name = f"{edge_type}_{s_type}_{t_type}"
                if s_type == node_type:
                    with contextlib.suppress(RuntimeError):
                        self._conn.execute(
                            f"MATCH (a:{s_type})-[r:{table_name}]->(b:{t_type}) "
                            f"WHERE a.id = $id DELETE r",
                            parameters={"id": node_id},
                        )
                if t_type == node_type:
                    with contextlib.suppress(RuntimeError):
                        self._conn.execute(
                            f"MATCH (a:{s_type})-[r:{table_name}]->(b:{t_type}) "
                            f"WHERE b.id = $id DELETE r",
                            parameters={"id": node_id},
                        )

    def delete_edges_for_file(self, file_path: str) -> None:
        """Delete all edges associated with nodes from a specific file."""
        nodes = self.get_nodes_for_file(file_path)
        for node in nodes:
            self._delete_edges_for_node(node.id, node.node_type)

    def get_file_hashes(self) -> dict[str, str]:
        """Get all stored file hashes from File nodes."""
        hashes: dict[str, str] = {}
        try:
            result = self._conn.execute("MATCH (f:File) RETURN f.path, f.hash")
            while result.has_next():
                row = result.get_next()
                path, hash_val = row[0], row[1]
                if path and hash_val:
                    hashes[path] = hash_val
        except RuntimeError:
            pass
        return hashes

    def get_nodes_for_file(self, file_path: str) -> list[NodeData]:
        """Get all nodes associated with a file."""
        nodes: list[NodeData] = []
        # Check node types that have a 'file' or 'path' property
        file_node_types = {
            "File": "path",
            "Class": "file",
            "Function": "file",
            "Method": "file",
            "Interface": "file",
            "Variable": "file",
        }
        for node_type, prop in file_node_types.items():
            try:
                result = self._conn.execute(
                    f"MATCH (n:{node_type}) WHERE n.{prop} = $path RETURN n.id",
                    parameters={"path": file_path},
                )
                while result.has_next():
                    row = result.get_next()
                    node = self.get_node(row[0])
                    if node:
                        nodes.append(node)
            except RuntimeError:
                continue
        return nodes

    def get_chunk_ids_for_file(self, file_path: str) -> list[str]:
        """Get IDs of all Chunk nodes linked to a file's nodes."""
        chunk_ids: list[str] = []
        # Find chunks via CHUNK_OF edges to nodes in this file
        file_nodes = self.get_nodes_for_file(file_path)
        for file_node in file_nodes:
            tgt_type = file_node.node_type
            table_name = f"CHUNK_OF_Chunk_{tgt_type}"
            try:
                result = self._conn.execute(
                    f"MATCH (c:Chunk)-[:{table_name}]->(n:{tgt_type}) WHERE n.id = $id RETURN c.id",
                    parameters={"id": file_node.id},
                )
                while result.has_next():
                    row = result.get_next()
                    chunk_ids.append(row[0])
            except RuntimeError:
                continue
        return chunk_ids

    def delete_nodes_for_file(self, file_path: str) -> None:
        """Delete all nodes and edges associated with a file."""
        # First delete chunks
        chunk_ids = self.get_chunk_ids_for_file(file_path)
        for chunk_id in chunk_ids:
            self.delete_node(chunk_id)

        # Then delete file nodes
        nodes = self.get_nodes_for_file(file_path)
        for node in nodes:
            self.delete_node(node.id)

    def clear(self) -> None:
        """Delete all nodes and edges from the graph."""
        for edge_type, endpoints in EDGE_ENDPOINTS.items():
            for s_type, t_type in endpoints:
                table_name = f"{edge_type}_{s_type}_{t_type}"
                with contextlib.suppress(RuntimeError):
                    self._conn.execute(
                        f"MATCH (a:{s_type})-[r:{table_name}]->(b:{t_type}) DELETE r"
                    )
        for node_type in NODE_TYPES:
            with contextlib.suppress(RuntimeError):
                self._conn.execute(f"MATCH (n:{node_type}) DELETE n")

    def close(self) -> None:
        """Close the database connection."""
        self._conn = None  # type: ignore[assignment]
        self._db = None  # type: ignore[assignment]
