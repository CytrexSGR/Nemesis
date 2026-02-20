"""Datenmodelle fuer die Indexing Pipeline."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class ChangeType(enum.Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class Chunk:
    id: str
    content: str
    token_count: int
    parent_node_id: str
    parent_type: str
    start_line: int
    end_line: int
    file_path: Path
    embedding_id: str | None = None


@dataclass
class FileChange:
    path: Path
    change_type: ChangeType
    old_hash: str | None
    new_hash: str | None


@dataclass
class IndexResult:
    files_indexed: int
    nodes_created: int
    edges_created: int
    chunks_created: int
    embeddings_created: int
    duration_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0
