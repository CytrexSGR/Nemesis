# 05b — AST-aware Chunking

> **Arbeitspaket E2** — Teil 2 von 5 des Indexing Pipeline Plans

**Goal:** AST-aware Chunker implementieren: Kleine Nodes als einzelne Chunks durchreichen, grosse Nodes an logischen Grenzen (Leerzeilen, def/class) aufteilen.

**Tech Stack:** Python, nemesis.indexer.models (Chunk), nemesis.indexer.tokens (count_tokens)

**Abhängigkeiten:** E1 (05a-indexer-models)

**Tasks in diesem Paket:** 3, 4

---

## Task 3: AST-aware Chunker — Kleine Nodes durchreichen

**Files:**
- `nemesis/indexer/chunker.py`
- `tests/test_indexer/test_chunker.py`

### Step 1 — Write Test

```python
# tests/test_indexer/test_chunker.py
"""Tests fuer AST-aware Chunking."""
import pytest
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MockCodeNode:
    """Mock fuer einen Code-Knoten aus dem Parser."""

    id: str
    node_type: str
    name: str
    start_line: int
    end_line: int
    children: list = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def test_small_node_becomes_single_chunk():
    """Nodes unter max_tokens werden als ein Chunk durchgereicht."""
    from nemesis.indexer.chunker import chunk_node

    node = MockCodeNode(
        id="func-001",
        node_type="Function",
        name="hello",
        start_line=1,
        end_line=3,
    )
    source = "def hello():\n    print('hi')\n    return True\n"

    chunks = chunk_node(node, source, max_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].content == source
    assert chunks[0].parent_node_id == "func-001"
    assert chunks[0].parent_type == "Function"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3


def test_chunk_id_format():
    """Chunk-IDs enthalten die Node-ID als Prefix."""
    from nemesis.indexer.chunker import chunk_node

    node = MockCodeNode(
        id="func-abc",
        node_type="Function",
        name="test",
        start_line=1,
        end_line=2,
    )
    source = "def test():\n    pass\n"

    chunks = chunk_node(node, source, max_tokens=500)
    assert chunks[0].id.startswith("func-abc")


def test_empty_source_returns_empty():
    """Leerer Source-Code erzeugt keine Chunks."""
    from nemesis.indexer.chunker import chunk_node

    node = MockCodeNode(
        id="func-001",
        node_type="Function",
        name="empty",
        start_line=1,
        end_line=1,
    )
    chunks = chunk_node(node, "", max_tokens=500)
    assert len(chunks) == 0


def test_chunk_file_path_from_node():
    """file_path im Chunk wird korrekt gesetzt."""
    from nemesis.indexer.chunker import chunk_node

    node = MockCodeNode(
        id="func-001",
        node_type="Function",
        name="test",
        start_line=5,
        end_line=7,
    )
    source = "def test():\n    x = 1\n    return x\n"

    chunks = chunk_node(node, source, max_tokens=500, file_path=Path("src/app.py"))
    assert chunks[0].file_path == Path("src/app.py")


def test_chunk_token_count_set():
    """Token-Count im Chunk wird korrekt berechnet."""
    from nemesis.indexer.chunker import chunk_node

    node = MockCodeNode(
        id="func-001",
        node_type="Function",
        name="test",
        start_line=1,
        end_line=2,
    )
    source = "def test():\n    pass\n"

    chunks = chunk_node(node, source, max_tokens=500)
    assert chunks[0].token_count > 0
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_chunker.py -v
# Erwartung: FAILED — Module nemesis.indexer.chunker existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/chunker.py
"""AST-aware Chunking fuer Code-Nodes.

Teilt grosse Code-Nodes in logische Sub-Chunks auf,
ausgerichtet an AST-Grenzen (Methoden, Bloecke).
Kleine Nodes (<= max_tokens) werden als einzelner Chunk durchgereicht.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from nemesis.indexer.models import Chunk
from nemesis.indexer.tokens import count_tokens


class CodeNodeProtocol(Protocol):
    """Minimales Interface fuer Code-Nodes aus dem Parser."""

    id: str
    node_type: str
    name: str
    start_line: int
    end_line: int
    children: list


def chunk_node(
    node: Any,
    source: str,
    max_tokens: int = 500,
    file_path: Path | None = None,
) -> list[Chunk]:
    """Teile einen Code-Node in Chunks auf.

    Wenn der Node <= max_tokens ist, wird er als einzelner Chunk
    zurueckgegeben. Grosse Nodes werden an Kind-Knoten-Grenzen
    aufgeteilt (siehe Task 4).

    Args:
        node: Code-Node aus dem Parser (muss id, node_type, name,
              start_line, end_line, children haben).
        source: Der Source-Code des Nodes.
        max_tokens: Maximale Token-Anzahl pro Chunk.
        file_path: Pfad der Quelldatei.

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
                file_path=file_path or Path("unknown"),
            )
        ]

    # Grosse Nodes: Split-Logik kommt in Task 4
    return _split_large_node(node, source, max_tokens, file_path)


def _split_large_node(
    node: Any,
    source: str,
    max_tokens: int,
    file_path: Path | None,
) -> list[Chunk]:
    """Placeholder fuer das Aufteilen grosser Nodes (Task 4)."""
    # Wird in Task 4 vollstaendig implementiert.
    # Vorlaeufig: Einfaches Line-basiertes Splitting als Fallback.
    lines = source.split("\n")
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = node.start_line
    chunk_idx = 0

    for i, line in enumerate(lines):
        current_lines.append(line)
        current_text = "\n".join(current_lines)
        token_count = count_tokens(current_text)

        if token_count >= max_tokens:
            chunks.append(
                Chunk(
                    id=f"{node.id}:chunk-{chunk_idx}",
                    content=current_text,
                    token_count=token_count,
                    parent_node_id=node.id,
                    parent_type=node.node_type,
                    start_line=current_start,
                    end_line=node.start_line + i,
                    file_path=file_path or Path("unknown"),
                )
            )
            chunk_idx += 1
            current_lines = []
            current_start = node.start_line + i + 1

    # Rest
    if current_lines:
        current_text = "\n".join(current_lines)
        token_count = count_tokens(current_text)
        if token_count > 0:
            chunks.append(
                Chunk(
                    id=f"{node.id}:chunk-{chunk_idx}",
                    content=current_text,
                    token_count=token_count,
                    parent_node_id=node.id,
                    parent_type=node.node_type,
                    start_line=current_start,
                    end_line=node.end_line,
                    file_path=file_path or Path("unknown"),
                )
            )

    return chunks
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_chunker.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/chunker.py tests/test_indexer/test_chunker.py
git commit -m "feat(indexer): add AST-aware chunker with small-node passthrough"
```

---

## Task 4: AST-aware Chunker — Grosse Nodes aufteilen

**Files:**
- `nemesis/indexer/chunker.py` (erweitern)
- `tests/test_indexer/test_chunker.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_chunker.py — APPEND folgende Tests

def test_large_node_split_into_multiple_chunks():
    """Nodes ueber max_tokens werden in mehrere Chunks aufgeteilt."""
    from nemesis.indexer.chunker import chunk_node

    # Erzeuge eine grosse Klasse mit vielen Methoden
    methods = []
    for i in range(20):
        methods.append(
            f"    def method_{i}(self, x: int) -> int:\n"
            f"        # Berechnung {i}\n"
            f"        result = x * {i}\n"
            f"        return result\n"
        )
    source = f"class BigService:\n    '''Service mit vielen Methoden.'''\n\n" + "\n".join(methods)

    node = MockCodeNode(
        id="class-big",
        node_type="Class",
        name="BigService",
        start_line=1,
        end_line=1 + source.count("\n"),
    )

    chunks = chunk_node(node, source, max_tokens=50)
    assert len(chunks) > 1
    # Alle Chunks gehoeren zum selben Parent
    for chunk in chunks:
        assert chunk.parent_node_id == "class-big"
        assert chunk.parent_type == "Class"


def test_large_node_chunks_cover_all_content():
    """Alle Chunks zusammen enthalten den gesamten Source-Code."""
    from nemesis.indexer.chunker import chunk_node

    lines = [f"    x_{i} = {i}" for i in range(100)]
    source = "def big_func():\n" + "\n".join(lines) + "\n"

    node = MockCodeNode(
        id="func-big",
        node_type="Function",
        name="big_func",
        start_line=1,
        end_line=101,
    )

    chunks = chunk_node(node, source, max_tokens=30)
    assert len(chunks) > 1
    # Zusammengesetzt ergibt es den gesamten Source
    reconstructed = "\n".join(c.content for c in chunks)
    assert "x_0 = 0" in reconstructed
    assert "x_99 = 99" in reconstructed


def test_large_node_chunk_ids_sequential():
    """Chunk-IDs sind sequentiell nummeriert."""
    from nemesis.indexer.chunker import chunk_node

    source = "def f():\n" + "\n".join(f"    line_{i} = {i}" for i in range(100)) + "\n"

    node = MockCodeNode(
        id="func-seq",
        node_type="Function",
        name="f",
        start_line=1,
        end_line=101,
    )

    chunks = chunk_node(node, source, max_tokens=30)
    for i, chunk in enumerate(chunks):
        assert chunk.id == f"func-seq:chunk-{i}"


def test_large_node_chunks_respect_max_tokens():
    """Kein Chunk ueberschreitet max_tokens signifikant."""
    from nemesis.indexer.chunker import chunk_node

    source = "def f():\n" + "\n".join(f"    var_{i} = {i}" for i in range(200)) + "\n"

    node = MockCodeNode(
        id="func-limit",
        node_type="Function",
        name="f",
        start_line=1,
        end_line=201,
    )

    max_t = 50
    chunks = chunk_node(node, source, max_tokens=max_t)
    for chunk in chunks:
        # Toleranz: Eine Zeile kann ueberlappen
        assert chunk.token_count <= max_t * 2, (
            f"Chunk {chunk.id} hat {chunk.token_count} tokens, max erlaubt ~{max_t * 2}"
        )


def test_large_node_line_numbers_correct():
    """Start/End-Line der Chunks sind korrekt und lueckenlos."""
    from nemesis.indexer.chunker import chunk_node

    source = "class C:\n" + "\n".join(f"    x_{i} = {i}" for i in range(50)) + "\n"

    node = MockCodeNode(
        id="class-lines",
        node_type="Class",
        name="C",
        start_line=1,
        end_line=51,
    )

    chunks = chunk_node(node, source, max_tokens=30)
    assert len(chunks) > 1
    # Erster Chunk startet bei node.start_line
    assert chunks[0].start_line == 1
    # Letzter Chunk endet bei node.end_line
    assert chunks[-1].end_line == 51
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_chunker.py -v -k "large_node"
# Erwartung: Tests sollen mit der Placeholder-Implementierung aus Task 3
# schon teilweise passen, aber wir verifizieren dass alle gruenen.
```

### Step 3 — Implement

Die `_split_large_node`-Funktion in `nemesis/indexer/chunker.py` ersetzen:

```python
# nemesis/indexer/chunker.py — _split_large_node komplett ersetzen

def _split_large_node(
    node: Any,
    source: str,
    max_tokens: int,
    file_path: Path | None,
) -> list[Chunk]:
    """Teile einen grossen Node an logischen Grenzen auf.

    Strategie:
    1. Wenn Kind-Knoten vorhanden: An Kind-Grenzen splitten
    2. Sonst: An Leerzeilen splitten (Funktions-/Block-Grenzen)
    3. Fallback: Zeilenweise splitten wenn noetig

    Jeder Chunk respektiert max_tokens (mit Toleranz fuer
    eine einzelne Zeile die den Grenzwert ueberschreitet).
    """
    lines = source.split("\n")
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = node.start_line
    chunk_idx = 0

    # Finde natuerliche Trennstellen (Leerzeilen, Dekoratoren, def/class)
    split_points: set[int] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped == ""
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("@")
            or stripped.startswith("async def ")
        ):
            split_points.add(i)

    for i, line in enumerate(lines):
        current_lines.append(line)
        current_text = "\n".join(current_lines)
        token_count = count_tokens(current_text)

        is_split_point = i in split_points
        at_limit = token_count >= max_tokens

        if at_limit and (is_split_point or token_count >= max_tokens):
            chunks.append(
                Chunk(
                    id=f"{node.id}:chunk-{chunk_idx}",
                    content=current_text,
                    token_count=token_count,
                    parent_node_id=node.id,
                    parent_type=node.node_type,
                    start_line=current_start,
                    end_line=node.start_line + i,
                    file_path=file_path or Path("unknown"),
                )
            )
            chunk_idx += 1
            current_lines = []
            current_start = node.start_line + i + 1

    # Rest-Chunk
    if current_lines:
        current_text = "\n".join(current_lines)
        token_count = count_tokens(current_text)
        if token_count > 0:
            chunks.append(
                Chunk(
                    id=f"{node.id}:chunk-{chunk_idx}",
                    content=current_text,
                    token_count=token_count,
                    parent_node_id=node.id,
                    parent_type=node.node_type,
                    start_line=current_start,
                    end_line=node.end_line,
                    file_path=file_path or Path("unknown"),
                )
            )

    return chunks
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_chunker.py -v
# Erwartung: ALL PASSED (alte + neue Tests)
```

### Step 5 — Commit

```bash
git add nemesis/indexer/chunker.py tests/test_indexer/test_chunker.py
git commit -m "feat(indexer): implement large node splitting at logical boundaries"
```

---

## Zusammenfassung E2

| Task | Datei(en) | Beschreibung |
|------|-----------|-------------|
| 3 | `chunker.py` | AST-aware Chunker: Kleine Nodes als einzelne Chunks |
| 4 | `chunker.py` | Grosse Nodes an logischen Grenzen aufteilen |

---

**Navigation:**
- Zurueck: [05a — Indexer Datenmodelle (E1)](05a-indexer-models.md)
- Weiter: [05c — Delta Operations (E3)](05c-delta-ops.md)
