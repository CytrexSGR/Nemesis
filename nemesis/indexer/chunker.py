"""AST-aware Chunking fuer Code-Knoten."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from nemesis.indexer.models import Chunk
from nemesis.indexer.tokens import count_tokens

if TYPE_CHECKING:
    from pathlib import Path

# Muster fuer natuerliche Splitpunkte in Python-Code
_SPLIT_PATTERN = re.compile(r"^\s*(def |class |async def |@)", re.MULTILINE)


def chunk_node(
    node,
    source: str,
    max_tokens: int = 500,
    file_path: Path | None = None,
) -> list[Chunk]:
    """Zerlegt einen AST-Knoten in Chunks.

    Kleine Knoten (<=max_tokens) werden als einzelner Chunk zurueckgegeben.
    Grosse Knoten werden an natuerlichen Grenzen aufgeteilt.

    Args:
        node: Ein Objekt mit id, node_type, name, start_line, end_line.
        source: Quellcode-Text des Knotens.
        max_tokens: Maximale Token-Anzahl pro Chunk.
        file_path: Dateipfad des Quellcodes.

    Returns:
        Liste von Chunk-Objekten.
    """
    if not source or not source.strip():
        return []

    token_count = count_tokens(source)

    if token_count <= max_tokens:
        return [
            Chunk(
                id=f"{node.id}:chunk-0",
                content=source,
                token_count=token_count,
                parent_node_id=node.id,
                parent_type=node.node_type,
                start_line=node.start_line,
                end_line=node.end_line,
                file_path=file_path,
            )
        ]

    return _split_large_node(node, source, max_tokens, file_path)


def _find_split_points(lines: list[str]) -> set[int]:
    """Findet natuerliche Splitpunkte: Leerzeilen und def/class/async def/@."""
    points: set[int] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _SPLIT_PATTERN.match(line):
            points.add(i)
    return points


def _split_large_node(
    node,
    source: str,
    max_tokens: int,
    file_path: Path | None,
) -> list[Chunk]:
    """Teilt einen grossen Knoten an natuerlichen Grenzen auf."""
    lines = source.splitlines(keepends=True)
    split_points = _find_split_points(lines)

    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = 0  # Index in lines (0-basiert)

    for i, line in enumerate(lines):
        current_lines.append(line)
        current_text = "".join(current_lines)
        current_tokens = count_tokens(current_text)

        # Splitten wenn Token-Limit erreicht und wir an einem Splitpunkt sind
        # oder wenn wir das Doppelte ueberschreiten (Notfall-Split)
        should_split = False
        if current_tokens >= max_tokens:
            if i in split_points or i + 1 in split_points:
                should_split = True
            elif current_tokens >= max_tokens:
                # Auch ohne natuerlichen Splitpunkt splitten
                should_split = True

        if should_split and i < len(lines) - 1:
            chunk_content = "".join(current_lines)
            chunk_tokens = count_tokens(chunk_content)
            chunk_idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{node.id}:chunk-{chunk_idx}",
                    content=chunk_content,
                    token_count=chunk_tokens,
                    parent_node_id=node.id,
                    parent_type=node.node_type,
                    start_line=node.start_line + current_start,
                    end_line=node.start_line + i,
                    file_path=file_path,
                )
            )
            current_lines = []
            current_start = i + 1

    # Rest-Chunk
    if current_lines:
        chunk_content = "".join(current_lines)
        chunk_tokens = count_tokens(chunk_content)
        chunk_idx = len(chunks)
        chunks.append(
            Chunk(
                id=f"{node.id}:chunk-{chunk_idx}",
                content=chunk_content,
                token_count=chunk_tokens,
                parent_node_id=node.id,
                parent_type=node.node_type,
                start_line=node.start_line + current_start,
                end_line=node.end_line,
                file_path=file_path,
            )
        )

    return chunks
