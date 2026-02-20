"""Neo4j graph adapter for Nemesis.

Optional backend using the neo4j Python driver. Requires a running
Neo4j instance. Install with: pip install nemesis-ai[neo4j]
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import neo4j as neo4j_driver
except ImportError:
    neo4j_driver = None  # type: ignore[assignment]

from nemesis.graph.adapter import (
    NODE_TYPES,
    EdgeData,
    NodeData,
    TraversalResult,
)

logger = logging.getLogger(__name__)


class Neo4jAdapter:
    """Neo4j graph database adapter.

    Requires a running Neo4j instance. Uses the official neo4j
    Python driver for communication.

    Args:
        uri: Neo4j connection URI (e.g. "bolt://localhost:7687").
        user: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name (default: "neo4j").
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ) -> None:
        if neo4j_driver is None:
            raise ImportError(
                "neo4j driver is required for Neo4jAdapter. "
                "Install with: pip install nemesis-ai[neo4j]"
            )
        self._driver = neo4j_driver.GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def _session(self):
        """Create a new database session."""
        return self._driver.session(database=self._database)

    def create_schema(self) -> None:
        """Create indexes and constraints for all node types.

        Neo4j uses labels instead of table DDL, so schema creation
        means creating uniqueness constraints and indexes.
        """
        with self._session() as session:
            for node_type in NODE_TYPES:
                try:
                    session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS "
                        f"FOR (n:{node_type}) REQUIRE n.id IS UNIQUE"
                    )
                except Exception as e:
                    logger.debug("Constraint for %s: %s", node_type, e)

            # Create indexes on commonly queried properties
            for node_type, prop in [
                ("File", "path"),
                ("Function", "file"),
                ("Class", "file"),
                ("Method", "file"),
                ("Variable", "file"),
            ]:
                try:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.{prop})")
                except Exception as e:
                    logger.debug("Index for %s.%s: %s", node_type, prop, e)

    def add_node(self, node: NodeData) -> None:
        """Add or update a node using MERGE."""
        props = {"id": node.id, **node.properties}
        set_clause = ", ".join(f"n.{k} = ${k}" for k in props if k != "id")
        query = f"MERGE (n:{node.node_type} {{id: $id}})"
        if set_clause:
            query += f" SET {set_clause}"

        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(query, **kw), **props)

    def add_edge(self, edge: EdgeData) -> None:
        """Add an edge between two nodes."""
        props_clause = ""
        params: dict[str, Any] = {
            "src_id": edge.source_id,
            "tgt_id": edge.target_id,
        }
        if edge.properties:
            prop_sets = ", ".join(f"{k}: ${k}" for k in edge.properties)
            props_clause = f" {{{prop_sets}}}"
            params.update(edge.properties)

        query = (
            f"MATCH (a {{id: $src_id}}), (b {{id: $tgt_id}}) "
            f"MERGE (a)-[:{edge.edge_type}{props_clause}]->(b)"
        )

        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(query, **kw), **params)

    def get_node(self, node_id: str) -> NodeData | None:
        """Retrieve a node by its ID."""
        query = "MATCH (n {id: $id}) RETURN labels(n) AS labels, properties(n) AS props"

        with self._session() as session:

            def _read(tx, **kwargs):
                result = tx.run(query, **kwargs)
                record = result.single()
                return record

            record = session.execute_read(_read, id=node_id)

        if record is None:
            return None

        data = record.data()
        labels = data.get("labels", [])
        props = data.get("props", {})
        node_type = labels[0] if labels else "Unknown"
        node_id_val = props.pop("id", node_id)

        return NodeData(id=node_id_val, node_type=node_type, properties=props)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[NodeData]:
        """Get neighboring nodes."""
        rel_pattern = f":{edge_type}" if edge_type else ""

        if direction == "outgoing":
            query = (
                f"MATCH (a {{id: $id}})-[{rel_pattern}]->(b) "
                f"RETURN labels(b) AS labels, properties(b) AS props"
            )
        elif direction == "incoming":
            query = (
                f"MATCH (a {{id: $id}})<-[{rel_pattern}]-(b) "
                f"RETURN labels(b) AS labels, properties(b) AS props"
            )
        else:  # both
            query = (
                f"MATCH (a {{id: $id}})-[{rel_pattern}]-(b) "
                f"RETURN DISTINCT labels(b) AS labels, properties(b) AS props"
            )

        neighbors: list[NodeData] = []
        with self._session() as session:

            def _read(tx, **kwargs):
                result = tx.run(query, **kwargs)
                return [r.data() for r in result]

            records = session.execute_read(_read, id=node_id)

        for data in records:
            labels = data.get("labels", [])
            props = data.get("props", {})
            nid = props.pop("id", "")
            node_type = labels[0] if labels else "Unknown"
            neighbors.append(NodeData(id=nid, node_type=node_type, properties=props))

        return neighbors

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> TraversalResult:
        """Traverse the graph from a starting node."""
        rel_filter = "|".join(edge_types) if edge_types else ""
        rel_pattern = f":{rel_filter}" if rel_filter else ""

        query = (
            f"MATCH path = (start {{id: $id}})-[{rel_pattern}*1..{max_depth}]->(end) "
            f"UNWIND nodes(path) AS n "
            f"WITH DISTINCT n "
            f"RETURN labels(n) AS labels, properties(n) AS props"
        )

        all_nodes: dict[str, NodeData] = {}

        # Always include the start node
        start_node = self.get_node(start_id)
        if start_node:
            all_nodes[start_id] = start_node

        with self._session() as session:

            def _read(tx, **kwargs):
                result = tx.run(query, **kwargs)
                return [r.data() for r in result]

            try:
                records = session.execute_read(_read, id=start_id)
                for data in records:
                    labels = data.get("labels", [])
                    props = data.get("props", {})
                    nid = props.pop("id", "")
                    node_type = labels[0] if labels else "Unknown"
                    if nid not in all_nodes:
                        all_nodes[nid] = NodeData(id=nid, node_type=node_type, properties=props)
            except Exception as e:
                logger.debug("Traversal query failed: %s", e)

        return TraversalResult(nodes=list(all_nodes.values()), edges=[])

    def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Execute a raw Cypher query."""
        with self._session() as session:

            def _read(tx, **kwargs):
                result = tx.run(cypher, **kwargs)
                return [dict(record) for record in result]

            return session.execute_read(_read, **(parameters or {}))

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        query = "MATCH (n {id: $id}) DETACH DELETE n"
        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(query, **kw), id=node_id)

    def delete_edges_for_file(self, file_path: str) -> None:
        """Delete all edges for nodes associated with a file."""
        query = "MATCH (n)-[r]-() WHERE n.file = $path OR n.path = $path DELETE r"
        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(query, **kw), path=file_path)

    def get_file_hashes(self) -> dict[str, str]:
        """Get all stored file hashes from File nodes."""
        query = "MATCH (f:File) WHERE f.hash IS NOT NULL RETURN f.path AS path, f.hash AS hash"
        rows = self.query(query)
        return {r["path"]: r["hash"] for r in rows if r.get("path") and r.get("hash")}

    def get_nodes_for_file(self, file_path: str) -> list[NodeData]:
        """Get all nodes associated with a file."""
        query = (
            "MATCH (n) "
            "WHERE n.file = $path OR n.path = $path "
            "RETURN labels(n) AS labels, properties(n) AS props"
        )
        nodes: list[NodeData] = []
        rows = self.query(query, parameters={"path": file_path})
        for data in rows:
            labels = data.get("labels", [])
            props = data.get("props", {})
            nid = props.pop("id", "")
            node_type = labels[0] if labels else "Unknown"
            nodes.append(NodeData(id=nid, node_type=node_type, properties=props))
        return nodes

    def get_chunk_ids_for_file(self, file_path: str) -> list[str]:
        """Get IDs of all Chunk nodes linked to a file's nodes."""
        query = (
            "MATCH (c:Chunk)-[:CHUNK_OF]->(n) "
            "WHERE n.file = $path OR n.path = $path "
            "RETURN c.id AS id"
        )
        rows = self.query(query, parameters={"path": file_path})
        return [r["id"] for r in rows]

    def delete_nodes_for_file(self, file_path: str) -> None:
        """Delete all nodes and edges associated with a file."""
        # Delete chunks first
        chunk_query = (
            "MATCH (c:Chunk)-[:CHUNK_OF]->(n) "
            "WHERE n.file = $path OR n.path = $path "
            "DETACH DELETE c"
        )
        node_query = "MATCH (n) WHERE n.file = $path OR n.path = $path DETACH DELETE n"
        with self._session() as session:
            session.execute_write(lambda tx, **kw: tx.run(chunk_query, **kw), path=file_path)
            session.execute_write(lambda tx, **kw: tx.run(node_query, **kw), path=file_path)

    def clear(self) -> None:
        """Delete all nodes and edges from the graph."""
        with self._session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None
