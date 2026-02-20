"""LanceDB-based vector store for Nemesis.

Stores code chunk embeddings and supports similarity search
with optional metadata filtering.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import lancedb
import pyarrow as pa


@dataclass(frozen=True)
class SearchResult:
    """A single result from a vector similarity search.

    Attributes:
        id: Unique identifier of the stored vector.
        text: Original text content of the chunk.
        score: Similarity score (higher = more similar).
        metadata: Additional metadata stored with the vector.
    """

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# LanceDB table name
_TABLE_NAME = "chunks"


class VectorStore:
    """LanceDB vector store for code chunk embeddings.

    Provides async add/search/delete operations backed by LanceDB.
    Uses a single table with a fixed schema determined at initialization.

    Args:
        path: Directory path for LanceDB storage.
        table_name: Name of the LanceDB table.
    """

    def __init__(
        self,
        path: str,
        table_name: str = _TABLE_NAME,
    ) -> None:
        self._path = path
        self._table_name = table_name
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.LanceTable | None = None
        self._dimensions: int | None = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the store has been initialized."""
        return self._initialized

    def _require_initialized(self) -> None:
        """Raise RuntimeError if the store is not initialized."""
        if not self._initialized or self._table is None:
            raise RuntimeError("VectorStore is not initialized. Call initialize() first.")

    async def initialize(self, dimensions: int) -> None:
        """Initialize the vector store with the given embedding dimensions.

        Creates the LanceDB database and table if they don't exist.
        If the table already exists, it is reused.

        Args:
            dimensions: Dimensionality of the embedding vectors.
        """
        self._dimensions = dimensions

        loop = asyncio.get_event_loop()
        self._db = await loop.run_in_executor(None, lambda: lancedb.connect(self._path))

        schema = pa.schema(
            [
                pa.field("id", pa.utf8()),
                pa.field("text", pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), list_size=dimensions)),
                pa.field("metadata", pa.utf8()),  # JSON-encoded
            ]
        )

        existing = await loop.run_in_executor(None, lambda: self._db.list_tables())
        existing_names = existing.tables if hasattr(existing, "tables") else list(existing)

        if self._table_name in existing_names:
            self._table = await loop.run_in_executor(
                None, lambda: self._db.open_table(self._table_name)
            )
        else:
            self._table = await loop.run_in_executor(
                None,
                lambda: self._db.create_table(self._table_name, schema=schema),
            )

        self._initialized = True

    async def count(self) -> int:
        """Return the number of vectors in the store.

        Returns:
            The total number of stored vectors.
        """
        if not self._initialized or self._table is None:
            return 0

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self._table.count_rows())
        return result

    async def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add vectors with their texts and metadata to the store.

        Args:
            ids: Unique identifiers for each vector.
            texts: Original text content for each vector.
            embeddings: Embedding vectors.
            metadata: Metadata dicts for each vector.

        Raises:
            ValueError: If input lists have different lengths.
            RuntimeError: If the store is not initialized.
        """
        self._require_initialized()

        lengths = {len(ids), len(texts), len(embeddings), len(metadata)}
        if len(lengths) > 1:
            raise ValueError(
                "ids, texts, embeddings, and metadata must have the same length. "
                f"Got lengths: ids={len(ids)}, texts={len(texts)}, "
                f"embeddings={len(embeddings)}, metadata={len(metadata)}"
            )

        if not ids:
            return

        rows = [
            {
                "id": id_,
                "text": text,
                "vector": embedding,
                "metadata": json.dumps(metadata_item),
            }
            for id_, text, embedding, metadata_item in zip(
                ids, texts, embeddings, metadata, strict=True
            )
        ]

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._table.add(rows))

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for the most similar vectors.

        Args:
            query_embedding: The query vector to search against.
            limit: Maximum number of results to return.
            filter: Optional metadata filter. Keys are metadata field names,
                    values are exact-match values. Filters are applied as
                    SQL LIKE clauses on the JSON metadata column.

        Returns:
            List of SearchResult ordered by similarity (best first).
        """
        self._require_initialized()

        loop = asyncio.get_event_loop()

        row_count = await self.count()
        if row_count == 0:
            return []

        def _search() -> list[dict]:
            query = self._table.search(query_embedding).limit(limit)

            if filter:
                # Build a WHERE clause that filters on the JSON metadata string.
                # Each key-value pair is matched as a substring in the JSON.
                conditions = []
                for key, value in filter.items():
                    # json.dumps handles proper quoting for strings and numbers
                    json_fragment = f'"{key}": {json.dumps(value)}'
                    escaped = json_fragment.replace("'", "''")
                    conditions.append(f"metadata LIKE '%{escaped}%'")
                where_clause = " AND ".join(conditions)
                query = query.where(where_clause)

            return query.to_list()

        raw_results = await loop.run_in_executor(None, _search)

        results = []
        for row in raw_results:
            meta = json.loads(row["metadata"]) if row.get("metadata") else {}
            results.append(
                SearchResult(
                    id=row["id"],
                    text=row["text"],
                    score=float(1.0 - row.get("_distance", 0.0)),
                    metadata=meta,
                )
            )

        return results

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete.
        """
        self._require_initialized()

        if not ids:
            return

        loop = asyncio.get_event_loop()

        # Build an IN clause for the delete filter
        id_list = ", ".join(f"'{id_}'" for id_ in ids)
        where = f"id IN ({id_list})"

        await loop.run_in_executor(None, lambda: self._table.delete(where))

    async def delete_by_file(self, file_path: str) -> None:
        """Delete all vectors associated with a specific file.

        Uses a LIKE clause on the JSON metadata column to match the
        file path.

        Args:
            file_path: The file path to match in metadata.
        """
        self._require_initialized()

        loop = asyncio.get_event_loop()
        # Build a LIKE pattern that matches the JSON fragment for the file key.
        # The metadata is stored as a JSON string, so we search for the
        # substring '"file": "value"' within it.
        json_fragment = f'"file": {json.dumps(file_path)}'
        escaped = json_fragment.replace("'", "''")
        where = f"metadata LIKE '%{escaped}%'"

        await loop.run_in_executor(None, lambda: self._table.delete(where))

    async def close(self) -> None:
        """Close the vector store and release resources."""
        self._table = None
        self._db = None
        self._initialized = False

    async def __aenter__(self) -> VectorStore:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit -- closes the store."""
        await self.close()
