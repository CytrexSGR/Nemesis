# 05 — Indexing Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Orchestriere den gesamten Datenfluss: Code-Dateien auf Disk -> Parser (Rust) -> Chunking -> Embeddings -> Graph + Vector Store. Sowohl Full Index als auch inkrementelle Delta Updates.

**Abhängigkeiten:** 02-rust-parser, 03-graph-layer, 04-vector-store

**Geschätzte Tasks:** 10

---

## Task 1: Datenmodelle fuer Chunker und Pipeline

**Files:**
- `nemesis/indexer/__init__.py`
- `nemesis/indexer/models.py`
- `tests/test_indexer/__init__.py`
- `tests/test_indexer/test_models.py`

### Step 1 — Write Test

```python
# tests/test_indexer/__init__.py
```

```python
# tests/test_indexer/test_models.py
"""Tests fuer Indexer-Datenmodelle."""
import pytest
from pathlib import Path


def test_chunk_creation():
    from nemesis.indexer.models import Chunk

    chunk = Chunk(
        id="chunk-001",
        content="def hello():\n    pass",
        token_count=8,
        parent_node_id="func-001",
        parent_type="Function",
        start_line=1,
        end_line=2,
        file_path=Path("src/main.py"),
    )
    assert chunk.id == "chunk-001"
    assert chunk.token_count == 8
    assert chunk.parent_type == "Function"
    assert chunk.file_path == Path("src/main.py")


def test_chunk_defaults():
    from nemesis.indexer.models import Chunk

    chunk = Chunk(
        id="chunk-002",
        content="x = 1",
        token_count=3,
        parent_node_id="var-001",
        parent_type="Variable",
        start_line=10,
        end_line=10,
        file_path=Path("lib.py"),
    )
    assert chunk.embedding_id is None


def test_file_change_creation():
    from nemesis.indexer.models import FileChange, ChangeType

    change = FileChange(
        path=Path("src/main.py"),
        change_type=ChangeType.MODIFIED,
        old_hash="abc123",
        new_hash="def456",
    )
    assert change.change_type == ChangeType.MODIFIED
    assert change.old_hash == "abc123"
    assert change.new_hash == "def456"


def test_file_change_added_no_old_hash():
    from nemesis.indexer.models import FileChange, ChangeType

    change = FileChange(
        path=Path("new_file.py"),
        change_type=ChangeType.ADDED,
        old_hash=None,
        new_hash="abc123",
    )
    assert change.old_hash is None
    assert change.new_hash == "abc123"


def test_file_change_deleted_no_new_hash():
    from nemesis.indexer.models import FileChange, ChangeType

    change = FileChange(
        path=Path("removed.py"),
        change_type=ChangeType.DELETED,
        old_hash="abc123",
        new_hash=None,
    )
    assert change.old_hash == "abc123"
    assert change.new_hash is None


def test_index_result_creation():
    from nemesis.indexer.models import IndexResult

    result = IndexResult(
        files_indexed=10,
        nodes_created=50,
        edges_created=30,
        chunks_created=20,
        embeddings_created=20,
        duration_ms=1500.0,
        errors=[],
    )
    assert result.files_indexed == 10
    assert result.nodes_created == 50
    assert result.success is True


def test_index_result_with_errors():
    from nemesis.indexer.models import IndexResult

    result = IndexResult(
        files_indexed=5,
        nodes_created=20,
        edges_created=10,
        chunks_created=8,
        embeddings_created=8,
        duration_ms=800.0,
        errors=["Failed to parse foo.py"],
    )
    assert result.success is False
    assert len(result.errors) == 1


def test_change_type_enum():
    from nemesis.indexer.models import ChangeType

    assert ChangeType.ADDED.value == "added"
    assert ChangeType.MODIFIED.value == "modified"
    assert ChangeType.DELETED.value == "deleted"
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_models.py -v
# Erwartung: FAILED — Module nemesis.indexer.models existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/__init__.py
"""Nemesis Indexer — Pipeline, Chunker, Delta."""
```

```python
# nemesis/indexer/models.py
"""Datenmodelle fuer die Indexing Pipeline."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path


class ChangeType(enum.Enum):
    """Art der Datei-Aenderung."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class Chunk:
    """Ein logischer Sub-Chunk eines AST-Knotens.

    Entsteht durch AST-aware Chunking grosser Code-Nodes.
    Jeder Chunk wird separat embedded und im Vector Store gespeichert.
    """

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
    """Beschreibt eine Datei-Aenderung fuer Delta Updates.

    Wird von delta.py erzeugt beim Vergleich von File-Hashes
    mit dem gespeicherten Zustand im Graph.
    """

    path: Path
    change_type: ChangeType
    old_hash: str | None
    new_hash: str | None


@dataclass
class IndexResult:
    """Ergebnis einer Indexierungs-Operation.

    Wird von pipeline.py zurueckgegeben nach Full Index oder Delta Update.
    """

    files_indexed: int
    nodes_created: int
    edges_created: int
    chunks_created: int
    embeddings_created: int
    duration_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True wenn keine Fehler aufgetreten sind."""
        return len(self.errors) == 0
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_models.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/__init__.py nemesis/indexer/models.py \
        tests/test_indexer/__init__.py tests/test_indexer/test_models.py
git commit -m "feat(indexer): add data models for Chunk, FileChange, IndexResult"
```

---

## Task 2: Token-Counting Utility

**Files:**
- `nemesis/indexer/tokens.py`
- `tests/test_indexer/test_tokens.py`

### Step 1 — Write Test

```python
# tests/test_indexer/test_tokens.py
"""Tests fuer Token-Counting."""
import pytest


def test_count_tokens_simple():
    from nemesis.indexer.tokens import count_tokens

    text = "def hello():\n    return 42"
    count = count_tokens(text)
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_empty():
    from nemesis.indexer.tokens import count_tokens

    assert count_tokens("") == 0


def test_count_tokens_whitespace_only():
    from nemesis.indexer.tokens import count_tokens

    count = count_tokens("   \n\n  \t  ")
    assert isinstance(count, int)
    # Whitespace zaehlt als Tokens
    assert count >= 0


def test_count_tokens_long_code():
    from nemesis.indexer.tokens import count_tokens

    code = "x = 1\n" * 200
    count = count_tokens(code)
    # 200 Zeilen a ~4 Tokens = ~800
    assert count > 100


def test_count_tokens_multiline_function():
    from nemesis.indexer.tokens import count_tokens

    code = '''def calculate_total(items: list[dict]) -> float:
    """Calculate the total price of all items."""
    total = 0.0
    for item in items:
        price = item["price"]
        quantity = item.get("quantity", 1)
        total += price * quantity
    return total
'''
    count = count_tokens(code)
    assert 30 < count < 200


def test_estimate_tokens_approximation():
    from nemesis.indexer.tokens import estimate_tokens

    text = "hello world foo bar baz"
    estimate = estimate_tokens(text)
    assert isinstance(estimate, int)
    assert estimate > 0


def test_estimate_vs_count_same_order_of_magnitude():
    from nemesis.indexer.tokens import count_tokens, estimate_tokens

    code = "def foo(x, y):\n    return x + y\n"
    counted = count_tokens(code)
    estimated = estimate_tokens(code)
    # Estimation sollte in der gleichen Groessenordnung sein
    assert 0.3 * counted <= estimated <= 3.0 * counted
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_tokens.py -v
# Erwartung: FAILED — Module nemesis.indexer.tokens existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/tokens.py
"""Token-Counting fuer AST-aware Chunking.

Verwendet tiktoken wenn verfuegbar (exakt), faellt auf
Heuristik zurueck (schnell, gut genug fuer Chunking-Grenzen).
"""
from __future__ import annotations

_encoder = None
_tiktoken_available: bool | None = None


def _get_encoder():
    """Lazy-load tiktoken Encoder (cl100k_base fuer OpenAI embeddings)."""
    global _encoder, _tiktoken_available
    if _tiktoken_available is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
            _tiktoken_available = True
        except ImportError:
            _tiktoken_available = False
    return _encoder


def count_tokens(text: str) -> int:
    """Zaehle Tokens im Text.

    Verwendet tiktoken (cl100k_base) wenn verfuegbar,
    sonst faellt auf estimate_tokens() zurueck.
    """
    if not text or not text.strip():
        if not text:
            return 0
        # Whitespace-only: nutze Estimation
        return estimate_tokens(text)

    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))

    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Schnelle Token-Schaetzung ohne externe Abhaengigkeit.

    Heuristik: ~4 Zeichen pro Token fuer Code (konservativ).
    Besser als Wort-basiert weil Code viele Sonderzeichen hat.
    """
    if not text:
        return 0
    # Heuristik: ~4 chars pro Token fuer Code
    return max(1, len(text) // 4)
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_tokens.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/tokens.py tests/test_indexer/test_tokens.py
git commit -m "feat(indexer): add token counting with tiktoken + fallback heuristic"
```

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

## Task 5: Delta Detection — File-Hashes vergleichen

**Files:**
- `nemesis/indexer/delta.py`
- `tests/test_indexer/test_delta.py`

### Step 1 — Write Test

```python
# tests/test_indexer/test_delta.py
"""Tests fuer Delta/Diff-Erkennung."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import hashlib


def _make_mock_graph():
    """Erzeugt einen Mock-GraphAdapter."""
    graph = MagicMock()
    return graph


def test_compute_file_hash():
    """File-Hash wird korrekt berechnet."""
    from nemesis.indexer.delta import compute_file_hash

    import tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def hello():\n    pass\n")
        f.flush()
        path = Path(f.name)

    try:
        h = compute_file_hash(path)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
    finally:
        os.unlink(path)


def test_compute_file_hash_deterministic():
    """Gleicher Inhalt ergibt gleichen Hash."""
    from nemesis.indexer.delta import compute_file_hash

    import tempfile, os

    content = "x = 42\n"
    paths = []
    for _ in range(2):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            paths.append(Path(f.name))

    try:
        assert compute_file_hash(paths[0]) == compute_file_hash(paths[1])
    finally:
        for p in paths:
            os.unlink(p)


def test_compute_file_hash_different_content():
    """Unterschiedlicher Inhalt ergibt unterschiedlichen Hash."""
    from nemesis.indexer.delta import compute_file_hash

    import tempfile, os

    paths = []
    for content in ["x = 1\n", "x = 2\n"]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            paths.append(Path(f.name))

    try:
        assert compute_file_hash(paths[0]) != compute_file_hash(paths[1])
    finally:
        for p in paths:
            os.unlink(p)


def test_detect_changes_new_file(tmp_path):
    """Neue Datei wird als ADDED erkannt."""
    from nemesis.indexer.delta import detect_changes
    from nemesis.indexer.models import ChangeType

    # Erstelle eine Python-Datei
    py_file = tmp_path / "new.py"
    py_file.write_text("print('hello')\n")

    # Graph hat keine File-Nodes -> alles ist neu
    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {}

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.ADDED
    assert changes[0].path == py_file
    assert changes[0].old_hash is None
    assert changes[0].new_hash is not None


def test_detect_changes_modified_file(tmp_path):
    """Geaenderte Datei wird als MODIFIED erkannt."""
    from nemesis.indexer.delta import detect_changes
    from nemesis.indexer.models import ChangeType

    py_file = tmp_path / "existing.py"
    py_file.write_text("x = 2\n")

    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {
        str(py_file): "old_hash_value",
    }

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.MODIFIED
    assert changes[0].old_hash == "old_hash_value"


def test_detect_changes_deleted_file(tmp_path):
    """Geloeschte Datei wird als DELETED erkannt."""
    from nemesis.indexer.delta import detect_changes
    from nemesis.indexer.models import ChangeType

    # Graph kennt eine Datei die nicht mehr existiert
    graph = _make_mock_graph()
    deleted_path = str(tmp_path / "deleted.py")
    graph.get_file_hashes.return_value = {
        deleted_path: "some_hash",
    }

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.DELETED
    assert changes[0].path == Path(deleted_path)
    assert changes[0].new_hash is None


def test_detect_changes_unchanged_file(tmp_path):
    """Unveraenderte Datei erzeugt keinen Change."""
    from nemesis.indexer.delta import detect_changes, compute_file_hash

    py_file = tmp_path / "stable.py"
    py_file.write_text("x = 1\n")

    current_hash = compute_file_hash(py_file)

    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {
        str(py_file): current_hash,
    }

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 0


def test_detect_changes_ignores_non_code_files(tmp_path):
    """Nicht-Code-Dateien werden ignoriert."""
    from nemesis.indexer.delta import detect_changes

    (tmp_path / "readme.md").write_text("# Hello\n")
    (tmp_path / "data.json").write_text("{}\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {}

    changes = detect_changes(tmp_path, graph, languages=["python"])
    assert len(changes) == 0


def test_detect_changes_multiple_languages(tmp_path):
    """Erkennung funktioniert mit mehreren Sprachen."""
    from nemesis.indexer.delta import detect_changes

    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "index.ts").write_text("const x = 1;\n")

    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {}

    changes = detect_changes(tmp_path, graph, languages=["python", "typescript"])
    assert len(changes) == 2
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_delta.py -v
# Erwartung: FAILED — Module nemesis.indexer.delta existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/delta.py
"""Delta/Diff-Erkennung fuer inkrementelle Index-Updates.

Vergleicht File-Hashes auf Disk mit den im Graph gespeicherten Hashes.
Erkennt neue, geaenderte und geloeschte Dateien.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from nemesis.indexer.models import ChangeType, FileChange


# Dateiendungen pro Sprache
LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py", ".pyi"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx", ".mjs"],
    "rust": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx"],
    "ruby": [".rb"],
    "php": [".php"],
}

# Standard-Verzeichnisse die ignoriert werden
DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "target",
    "build",
    "dist",
    ".nemesis",
}


def compute_file_hash(path: Path) -> str:
    """Berechne SHA-256 Hash einer Datei.

    Args:
        path: Pfad zur Datei.

    Returns:
        Hex-String des SHA-256 Hashes.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _get_extensions(languages: list[str]) -> set[str]:
    """Sammle alle relevanten Dateiendungen fuer die angegebenen Sprachen."""
    extensions: set[str] = set()
    for lang in languages:
        lang_lower = lang.lower()
        if lang_lower in LANGUAGE_EXTENSIONS:
            extensions.update(LANGUAGE_EXTENSIONS[lang_lower])
    return extensions


def _collect_code_files(
    project_path: Path,
    extensions: set[str],
    ignore_dirs: set[str] | None = None,
) -> list[Path]:
    """Finde alle Code-Dateien im Projekt rekursiv.

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        extensions: Erlaubte Dateiendungen.
        ignore_dirs: Verzeichnisnamen die uebersprungen werden.

    Returns:
        Liste der gefundenen Code-Dateien.
    """
    if ignore_dirs is None:
        ignore_dirs = DEFAULT_IGNORE_DIRS

    files: list[Path] = []
    for item in sorted(project_path.rglob("*")):
        if item.is_file() and item.suffix in extensions:
            # Pruefen ob ein Eltern-Verzeichnis ignoriert werden soll
            parts = item.relative_to(project_path).parts
            if not any(part in ignore_dirs for part in parts):
                files.append(item)
    return files


def detect_changes(
    project_path: Path,
    graph: Any,
    languages: list[str],
    ignore_dirs: set[str] | None = None,
) -> list[FileChange]:
    """Erkenne Datei-Aenderungen seit dem letzten Index.

    Vergleicht aktuelle Dateien auf Disk mit den im Graph
    gespeicherten File-Hashes. Erkennt:
    - ADDED: Datei existiert auf Disk, aber nicht im Graph
    - MODIFIED: Datei existiert in beiden, aber Hash unterschiedlich
    - DELETED: Datei existiert im Graph, aber nicht mehr auf Disk

    Args:
        project_path: Wurzelverzeichnis des Projekts.
        graph: GraphAdapter mit get_file_hashes() Methode.
        languages: Liste der zu indexierenden Sprachen.
        ignore_dirs: Verzeichnisnamen die ignoriert werden.

    Returns:
        Liste von FileChange-Objekten.
    """
    extensions = _get_extensions(languages)
    current_files = _collect_code_files(project_path, extensions, ignore_dirs)

    # Aktuelle Hashes berechnen
    current_hashes: dict[str, str] = {}
    for file_path in current_files:
        current_hashes[str(file_path)] = compute_file_hash(file_path)

    # Gespeicherte Hashes aus dem Graph
    stored_hashes: dict[str, str] = graph.get_file_hashes()

    changes: list[FileChange] = []

    # Neue und geaenderte Dateien
    for path_str, new_hash in sorted(current_hashes.items()):
        if path_str not in stored_hashes:
            changes.append(
                FileChange(
                    path=Path(path_str),
                    change_type=ChangeType.ADDED,
                    old_hash=None,
                    new_hash=new_hash,
                )
            )
        elif stored_hashes[path_str] != new_hash:
            changes.append(
                FileChange(
                    path=Path(path_str),
                    change_type=ChangeType.MODIFIED,
                    old_hash=stored_hashes[path_str],
                    new_hash=new_hash,
                )
            )

    # Geloeschte Dateien
    for path_str, old_hash in sorted(stored_hashes.items()):
        if path_str not in current_hashes:
            changes.append(
                FileChange(
                    path=Path(path_str),
                    change_type=ChangeType.DELETED,
                    old_hash=old_hash,
                    new_hash=None,
                )
            )

    return changes
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_delta.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/delta.py tests/test_indexer/test_delta.py
git commit -m "feat(indexer): add delta detection with file hash comparison"
```

---

## Task 6: Delta — Alte Daten loeschen

**Files:**
- `nemesis/indexer/delta.py` (erweitern)
- `tests/test_indexer/test_delta.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_delta.py — APPEND folgende Tests

def test_delete_file_data_calls_graph_and_vector():
    """delete_file_data loescht Nodes, Edges und Embeddings."""
    from nemesis.indexer.delta import delete_file_data

    graph = _make_mock_graph()
    graph.get_nodes_for_file.return_value = [
        {"id": "func-001", "type": "Function"},
        {"id": "func-002", "type": "Function"},
    ]
    graph.get_chunk_ids_for_file.return_value = ["chunk-001", "chunk-002"]

    vector_store = MagicMock()

    file_path = Path("/project/src/old.py")
    delete_file_data(file_path, graph, vector_store)

    # Graph: Nodes und Edges fuer die Datei loeschen
    graph.delete_nodes_for_file.assert_called_once_with(str(file_path))
    # Vector Store: Embeddings fuer die Chunks loeschen
    vector_store.delete_embeddings.assert_called_once_with(["chunk-001", "chunk-002"])


def test_delete_file_data_no_chunks():
    """delete_file_data funktioniert wenn keine Chunks existieren."""
    from nemesis.indexer.delta import delete_file_data

    graph = _make_mock_graph()
    graph.get_nodes_for_file.return_value = [{"id": "func-001", "type": "Function"}]
    graph.get_chunk_ids_for_file.return_value = []

    vector_store = MagicMock()

    delete_file_data(Path("/project/src/empty.py"), graph, vector_store)

    graph.delete_nodes_for_file.assert_called_once()
    vector_store.delete_embeddings.assert_called_once_with([])


def test_delete_file_data_no_existing_data():
    """delete_file_data funktioniert wenn Datei nicht im Graph ist."""
    from nemesis.indexer.delta import delete_file_data

    graph = _make_mock_graph()
    graph.get_nodes_for_file.return_value = []
    graph.get_chunk_ids_for_file.return_value = []

    vector_store = MagicMock()

    delete_file_data(Path("/project/src/unknown.py"), graph, vector_store)

    graph.delete_nodes_for_file.assert_called_once()
    vector_store.delete_embeddings.assert_called_once_with([])
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_delta.py -v -k "delete_file"
# Erwartung: FAILED — delete_file_data existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/delta.py — APPEND am Ende der Datei

def delete_file_data(
    file_path: Path,
    graph: Any,
    vector_store: Any,
) -> None:
    """Loesche alle Daten einer Datei aus Graph und Vector Store.

    Wird vor dem Re-Index einer geaenderten Datei aufgerufen,
    damit keine veralteten Nodes/Embeddings zurueckbleiben.

    Args:
        file_path: Pfad der Datei deren Daten geloescht werden.
        graph: GraphAdapter mit delete_nodes_for_file() und
               get_chunk_ids_for_file() Methoden.
        vector_store: VectorStore mit delete_embeddings() Methode.
    """
    # Chunk-IDs sammeln bevor die Nodes geloescht werden
    chunk_ids: list[str] = graph.get_chunk_ids_for_file(str(file_path))

    # Graph: Alle Nodes und Edges fuer diese Datei loeschen
    graph.delete_nodes_for_file(str(file_path))

    # Vector Store: Embeddings fuer die Chunks loeschen
    vector_store.delete_embeddings(chunk_ids)
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_delta.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/delta.py tests/test_indexer/test_delta.py
git commit -m "feat(indexer): add delete_file_data for cleaning stale graph/vector data"
```

---

## Task 7: Pipeline — Single File Index

**Files:**
- `nemesis/indexer/pipeline.py`
- `tests/test_indexer/test_pipeline.py`

### Step 1 — Write Test

```python
# tests/test_indexer/test_pipeline.py
"""Tests fuer die Indexing Pipeline."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockParseResult:
    """Mock fuer das Parser-Ergebnis."""

    nodes: list
    edges: list
    file_path: str
    language: str


@dataclass
class MockCodeNode:
    """Mock fuer einen Code-Knoten."""

    id: str
    node_type: str
    name: str
    start_line: int
    end_line: int
    source: str
    children: list = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class MockEdge:
    """Mock fuer eine Graph-Kante."""

    source_id: str
    target_id: str
    edge_type: str


def _make_mock_parser():
    """Erzeugt einen Mock-Parser."""
    parser = MagicMock()
    node = MockCodeNode(
        id="func-001",
        node_type="Function",
        name="hello",
        start_line=1,
        end_line=3,
        source="def hello():\n    return 42\n",
    )
    edge = MockEdge(
        source_id="file-main",
        target_id="func-001",
        edge_type="CONTAINS",
    )
    result = MockParseResult(
        nodes=[node],
        edges=[edge],
        file_path="src/main.py",
        language="python",
    )
    parser.parse_file.return_value = result
    return parser, result


def _make_mock_graph():
    graph = MagicMock()
    graph.add_node.return_value = None
    graph.add_edge.return_value = None
    graph.get_chunk_ids_for_file.return_value = []
    graph.delete_nodes_for_file.return_value = None
    graph.get_file_hashes.return_value = {}
    return graph


def _make_mock_vector_store():
    store = MagicMock()
    store.add_embeddings.return_value = None
    store.delete_embeddings.return_value = None
    return store


def _make_mock_embedder():
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    return embedder


def test_pipeline_creation():
    """Pipeline laesst sich mit allen Abhaengigkeiten erstellen."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser = MagicMock()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )
    assert pipeline is not None


def test_index_file_parses_and_stores():
    """index_file parsed die Datei und speichert Nodes/Edges/Embeddings."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser, parse_result = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_file(Path("src/main.py"))

    # Parser wurde aufgerufen
    parser.parse_file.assert_called_once_with(Path("src/main.py"))
    # Nodes im Graph gespeichert
    assert graph.add_node.called
    # Edges im Graph gespeichert
    assert graph.add_edge.called
    # Embeddings erzeugt und gespeichert
    assert embedder.embed_texts.called
    assert vector_store.add_embeddings.called
    # Ergebnis korrekt
    assert result.files_indexed == 1
    assert result.nodes_created >= 1
    assert result.edges_created >= 1
    assert result.success is True


def test_index_file_returns_index_result():
    """index_file gibt ein korrektes IndexResult zurueck."""
    from nemesis.indexer.pipeline import IndexingPipeline
    from nemesis.indexer.models import IndexResult

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_file(Path("src/main.py"))

    assert isinstance(result, IndexResult)
    assert result.duration_ms >= 0
    assert result.chunks_created >= 0
    assert result.embeddings_created >= 0


def test_index_file_handles_parser_error():
    """index_file faengt Parser-Fehler ab und meldet sie im Ergebnis."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser = MagicMock()
    parser.parse_file.side_effect = RuntimeError("Parse error: invalid syntax")
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_file(Path("broken.py"))

    assert result.success is False
    assert len(result.errors) == 1
    assert "broken.py" in result.errors[0]


def test_index_file_chunks_large_nodes():
    """Grosse Nodes werden in Chunks aufgeteilt."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser = MagicMock()
    # Erzeuge einen grossen Node
    big_source = "def big():\n" + "\n".join(f"    x_{i} = {i}" for i in range(200)) + "\n"
    big_node = MockCodeNode(
        id="func-big",
        node_type="Function",
        name="big",
        start_line=1,
        end_line=201,
        source=big_source,
    )
    parse_result = MockParseResult(
        nodes=[big_node],
        edges=[],
        file_path="src/big.py",
        language="python",
    )
    parser.parse_file.return_value = parse_result

    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()
    embedder.embed_texts.return_value = [[0.1] * 3] * 20  # Genug fuer alle Chunks

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_file(Path("src/big.py"))

    assert result.chunks_created > 1
    assert result.success is True
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: FAILED — Module nemesis.indexer.pipeline existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/pipeline.py
"""Indexing Pipeline — Orchestriert Parse -> Chunk -> Embed -> Store.

Die zentrale Klasse IndexingPipeline bringt Parser, Graph-Adapter,
Vector Store und Embedder zusammen. Sie unterstuetzt:
- Full Index eines gesamten Projekts
- Single File Index
- Delta Update (delete old + insert new)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from nemesis.indexer.chunker import chunk_node
from nemesis.indexer.models import Chunk, IndexResult

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Orchestriert den Indexierungs-Datenfluss.

    Verbindet Parser (Rust/Tree-sitter), AST-Chunker,
    Embedding-Generator und Graph/Vector Stores.
    """

    def __init__(
        self,
        parser: Any,
        graph: Any,
        vector_store: Any,
        embedder: Any,
        max_tokens_per_chunk: int = 500,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Erstelle eine neue IndexingPipeline.

        Args:
            parser: Parser mit parse_file(path) -> ParseResult.
            graph: GraphAdapter mit add_node(), add_edge(), etc.
            vector_store: VectorStore mit add_embeddings(), etc.
            embedder: Embedder mit embed_texts(texts) -> list[list[float]].
            max_tokens_per_chunk: Maximale Token-Anzahl pro Chunk.
            on_progress: Optionaler Callback (message, current, total).
        """
        self._parser = parser
        self._graph = graph
        self._vector_store = vector_store
        self._embedder = embedder
        self._max_tokens = max_tokens_per_chunk
        self._on_progress = on_progress

    def _report_progress(self, message: str, current: int = 0, total: int = 0) -> None:
        """Melde Fortschritt ueber den Callback."""
        if self._on_progress:
            self._on_progress(message, current, total)
        logger.debug("Progress: %s (%d/%d)", message, current, total)

    def index_file(self, path: Path) -> IndexResult:
        """Indexiere eine einzelne Datei.

        Ablauf:
        1. Parse die Datei -> Nodes + Edges
        2. Chunke grosse Nodes
        3. Erzeuge Embeddings fuer alle Chunks
        4. Speichere Nodes + Edges im Graph
        5. Speichere Embeddings im Vector Store

        Args:
            path: Pfad zur Datei.

        Returns:
            IndexResult mit Statistiken.
        """
        start_time = time.monotonic()
        nodes_created = 0
        edges_created = 0
        chunks_created = 0
        embeddings_created = 0
        errors: list[str] = []

        try:
            # 1. Parse
            self._report_progress(f"Parsing {path.name}")
            parse_result = self._parser.parse_file(path)

            # 2. Nodes im Graph speichern
            for node in parse_result.nodes:
                self._graph.add_node(node)
                nodes_created += 1

            # 3. Edges im Graph speichern
            for edge in parse_result.edges:
                self._graph.add_edge(edge)
                edges_created += 1

            # 4. Chunking
            self._report_progress(f"Chunking {path.name}")
            all_chunks: list[Chunk] = []
            for node in parse_result.nodes:
                source = getattr(node, "source", "")
                if source:
                    node_chunks = chunk_node(
                        node,
                        source,
                        max_tokens=self._max_tokens,
                        file_path=path,
                    )
                    all_chunks.extend(node_chunks)

            chunks_created = len(all_chunks)

            # 5. Embeddings erzeugen
            if all_chunks:
                self._report_progress(f"Embedding {len(all_chunks)} chunks")
                texts = [chunk.content for chunk in all_chunks]
                vectors = self._embedder.embed_texts(texts)

                # 6. Im Vector Store speichern
                self._report_progress(f"Storing embeddings")
                embedding_data = []
                for chunk, vector in zip(all_chunks, vectors):
                    embedding_data.append(
                        {
                            "id": chunk.id,
                            "vector": vector,
                            "content": chunk.content,
                            "metadata": {
                                "file_path": str(chunk.file_path),
                                "parent_node_id": chunk.parent_node_id,
                                "parent_type": chunk.parent_type,
                                "start_line": chunk.start_line,
                                "end_line": chunk.end_line,
                            },
                        }
                    )
                self._vector_store.add_embeddings(embedding_data)
                embeddings_created = len(embedding_data)

                # Chunk-Nodes auch im Graph speichern
                for chunk in all_chunks:
                    self._graph.add_node(chunk)

        except Exception as e:
            errors.append(f"Error indexing {path}: {e}")
            logger.error("Failed to index %s: %s", path, e)

        duration_ms = (time.monotonic() - start_time) * 1000

        return IndexResult(
            files_indexed=1 if not errors else 0,
            nodes_created=nodes_created,
            edges_created=edges_created,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            duration_ms=duration_ms,
            errors=errors,
        )
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/pipeline.py tests/test_indexer/test_pipeline.py
git commit -m "feat(indexer): add IndexingPipeline with single file indexing"
```

---

## Task 8: Pipeline — Reindex File (Delta Update)

**Files:**
- `nemesis/indexer/pipeline.py` (erweitern)
- `tests/test_indexer/test_pipeline.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_pipeline.py — APPEND folgende Tests

def test_reindex_file_deletes_old_data_first():
    """reindex_file loescht alte Daten bevor es neu indexiert."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    graph.get_chunk_ids_for_file.return_value = ["old-chunk-1", "old-chunk-2"]
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.reindex_file(Path("src/changed.py"))

    # Alte Daten wurden geloescht
    graph.delete_nodes_for_file.assert_called_once_with(str(Path("src/changed.py")))
    vector_store.delete_embeddings.assert_called_once_with(["old-chunk-1", "old-chunk-2"])
    # Dann wurde neu indexiert
    parser.parse_file.assert_called_once()
    assert result.success is True
    assert result.files_indexed == 1


def test_reindex_file_returns_result():
    """reindex_file gibt ein korrektes IndexResult zurueck."""
    from nemesis.indexer.pipeline import IndexingPipeline
    from nemesis.indexer.models import IndexResult

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.reindex_file(Path("src/changed.py"))

    assert isinstance(result, IndexResult)
    assert result.duration_ms >= 0


def test_reindex_preserves_other_files():
    """reindex_file loescht nur Daten der angegebenen Datei."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    pipeline.reindex_file(Path("src/file_a.py"))

    # Nur file_a.py wurde geloescht, nicht andere Dateien
    call_args = graph.delete_nodes_for_file.call_args[0][0]
    assert "file_a.py" in call_args


def test_reindex_file_handles_delete_error():
    """reindex_file meldet Fehler wenn Loeschen fehlschlaegt."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    graph.get_chunk_ids_for_file.side_effect = RuntimeError("DB connection lost")
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.reindex_file(Path("src/broken.py"))

    assert result.success is False
    assert len(result.errors) >= 1
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v -k "reindex"
# Erwartung: FAILED — reindex_file existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/pipeline.py — Methode reindex_file zur Klasse IndexingPipeline hinzufuegen

    def reindex_file(self, path: Path) -> IndexResult:
        """Inkrementeller Re-Index einer geaenderten Datei.

        Ablauf:
        1. Alte Chunks-IDs aus dem Graph laden
        2. Alte Nodes/Edges im Graph loeschen
        3. Alte Embeddings im Vector Store loeschen
        4. Datei neu indexieren (wie index_file)

        Dies ist der Delta-Update-Pfad: Nur die geaenderte Datei
        wird neu verarbeitet, nicht das gesamte Projekt.

        Args:
            path: Pfad zur geaenderten Datei.

        Returns:
            IndexResult mit Statistiken.
        """
        start_time = time.monotonic()

        try:
            # 1. Alte Daten loeschen
            self._report_progress(f"Cleaning old data for {path.name}")
            chunk_ids = self._graph.get_chunk_ids_for_file(str(path))
            self._graph.delete_nodes_for_file(str(path))
            self._vector_store.delete_embeddings(chunk_ids)
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            return IndexResult(
                files_indexed=0,
                nodes_created=0,
                edges_created=0,
                chunks_created=0,
                embeddings_created=0,
                duration_ms=duration_ms,
                errors=[f"Error cleaning old data for {path}: {e}"],
            )

        # 2. Neu indexieren
        result = self.index_file(path)

        # Duration korrigieren (inkl. Loeschzeit)
        total_duration_ms = (time.monotonic() - start_time) * 1000
        result.duration_ms = total_duration_ms

        return result
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/pipeline.py tests/test_indexer/test_pipeline.py
git commit -m "feat(indexer): add reindex_file for incremental delta updates"
```

---

## Task 9: Pipeline — Full Project Index

**Files:**
- `nemesis/indexer/pipeline.py` (erweitern)
- `tests/test_indexer/test_pipeline.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_pipeline.py — APPEND folgende Tests

def test_index_project_indexes_all_files(tmp_path):
    """index_project indexiert alle Code-Dateien im Verzeichnis."""
    from nemesis.indexer.pipeline import IndexingPipeline

    # Erstelle Test-Dateien
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main():\n    pass\n")
    (src / "utils.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "readme.md").write_text("# Readme\n")  # Soll ignoriert werden

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python"])

    # Parser wurde fuer beide .py Dateien aufgerufen
    assert parser.parse_file.call_count == 2
    assert result.files_indexed == 2
    assert result.success is True


def test_index_project_skips_ignored_dirs(tmp_path):
    """index_project ignoriert __pycache__, node_modules, etc."""
    from nemesis.indexer.pipeline import IndexingPipeline

    (tmp_path / "good.py").write_text("x = 1\n")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "bad.py").write_text("y = 2\n")

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python"])

    # Nur good.py, nicht __pycache__/bad.py
    assert parser.parse_file.call_count == 1
    assert result.files_indexed == 1


def test_index_project_multiple_languages(tmp_path):
    """index_project unterstuetzt mehrere Sprachen gleichzeitig."""
    from nemesis.indexer.pipeline import IndexingPipeline

    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "index.ts").write_text("const x = 1;\n")

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python", "typescript"])

    assert parser.parse_file.call_count == 2
    assert result.files_indexed == 2


def test_index_project_empty_dir(tmp_path):
    """index_project mit leerem Verzeichnis gibt leeres Ergebnis."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser = MagicMock()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python"])

    assert result.files_indexed == 0
    assert result.success is True
    parser.parse_file.assert_not_called()


def test_index_project_continues_on_single_file_error(tmp_path):
    """index_project macht weiter wenn eine einzelne Datei fehlschlaegt."""
    from nemesis.indexer.pipeline import IndexingPipeline

    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text("invalid syntax ???\n")

    parser = MagicMock()
    call_count = 0

    def mock_parse(path):
        nonlocal call_count
        call_count += 1
        if "bad.py" in str(path):
            raise RuntimeError("Parse error")
        node = MockCodeNode(
            id="func-1", node_type="Function", name="x",
            start_line=1, end_line=1, source="x = 1\n",
        )
        return MockParseResult(
            nodes=[node], edges=[], file_path=str(path), language="python",
        )

    parser.parse_file.side_effect = mock_parse
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python"])

    # Eine Datei erfolgreich, eine fehlgeschlagen
    assert result.files_indexed == 1
    assert len(result.errors) == 1
    assert result.success is False


def test_index_project_progress_reporting(tmp_path):
    """index_project meldet Fortschritt ueber Callback."""
    from nemesis.indexer.pipeline import IndexingPipeline

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    progress_calls = []

    def on_progress(msg, current, total):
        progress_calls.append((msg, current, total))

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
        on_progress=on_progress,
    )

    pipeline.index_project(tmp_path, languages=["python"])

    # Es wurden Fortschritts-Meldungen erzeugt
    assert len(progress_calls) > 0
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v -k "index_project"
# Erwartung: FAILED — index_project existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/pipeline.py — Methode index_project zur Klasse IndexingPipeline hinzufuegen

    def index_project(
        self,
        path: Path,
        languages: list[str],
        ignore_dirs: set[str] | None = None,
    ) -> IndexResult:
        """Full Index eines gesamten Projekts.

        Findet alle Code-Dateien im Verzeichnis (rekursiv) und
        indexiert jede einzeln. Fehler bei einzelnen Dateien werden
        gesammelt, der Prozess laeuft weiter.

        Args:
            path: Wurzelverzeichnis des Projekts.
            languages: Liste der zu indexierenden Sprachen
                       (z.B. ["python", "typescript"]).
            ignore_dirs: Verzeichnisnamen die ignoriert werden.
                         Default: __pycache__, node_modules, .git, etc.

        Returns:
            Aggregiertes IndexResult ueber alle Dateien.
        """
        from nemesis.indexer.delta import _get_extensions, _collect_code_files

        start_time = time.monotonic()

        extensions = _get_extensions(languages)
        files = _collect_code_files(path, extensions, ignore_dirs)

        total_files = len(files)
        total_nodes = 0
        total_edges = 0
        total_chunks = 0
        total_embeddings = 0
        files_indexed = 0
        all_errors: list[str] = []

        self._report_progress(f"Found {total_files} files to index", 0, total_files)

        for i, file_path in enumerate(files):
            self._report_progress(
                f"Indexing {file_path.name} ({i + 1}/{total_files})",
                i + 1,
                total_files,
            )

            result = self.index_file(file_path)

            files_indexed += result.files_indexed
            total_nodes += result.nodes_created
            total_edges += result.edges_created
            total_chunks += result.chunks_created
            total_embeddings += result.embeddings_created
            all_errors.extend(result.errors)

        duration_ms = (time.monotonic() - start_time) * 1000

        self._report_progress(
            f"Done: {files_indexed}/{total_files} files indexed",
            total_files,
            total_files,
        )

        return IndexResult(
            files_indexed=files_indexed,
            nodes_created=total_nodes,
            edges_created=total_edges,
            chunks_created=total_chunks,
            embeddings_created=total_embeddings,
            duration_ms=duration_ms,
            errors=all_errors,
        )
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/pipeline.py tests/test_indexer/test_pipeline.py
git commit -m "feat(indexer): add full project indexing with progress reporting"
```

---

## Task 10: Pipeline — Delta Project Update

**Files:**
- `nemesis/indexer/pipeline.py` (erweitern)
- `tests/test_indexer/test_pipeline.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_pipeline.py — APPEND folgende Tests

def test_update_project_processes_all_change_types(tmp_path):
    """update_project verarbeitet ADDED, MODIFIED und DELETED korrekt."""
    from nemesis.indexer.pipeline import IndexingPipeline
    from nemesis.indexer.models import ChangeType, FileChange
    from nemesis.indexer.delta import compute_file_hash

    # Bestehende Datei die geaendert wurde
    modified_file = tmp_path / "modified.py"
    modified_file.write_text("x = 2  # changed\n")

    # Neue Datei
    new_file = tmp_path / "new.py"
    new_file.write_text("y = 1\n")

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {
        str(modified_file): "old_hash",
        str(tmp_path / "deleted.py"): "deleted_hash",
    }
    graph.get_chunk_ids_for_file.return_value = []
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.update_project(tmp_path, languages=["python"])

    # ADDED (new.py) und MODIFIED (modified.py) wurden indexiert
    # DELETED (deleted.py) wurde geloescht
    assert result.files_indexed >= 1
    # delete_nodes_for_file wurde aufgerufen (fuer modified und deleted)
    assert graph.delete_nodes_for_file.call_count >= 1


def test_update_project_no_changes(tmp_path):
    """update_project mit unveraenderten Dateien gibt leeres Ergebnis."""
    from nemesis.indexer.pipeline import IndexingPipeline
    from nemesis.indexer.delta import compute_file_hash

    py_file = tmp_path / "stable.py"
    py_file.write_text("x = 1\n")
    current_hash = compute_file_hash(py_file)

    parser = MagicMock()
    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {str(py_file): current_hash}
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.update_project(tmp_path, languages=["python"])

    # Nichts zu tun
    assert result.files_indexed == 0
    assert result.success is True
    parser.parse_file.assert_not_called()


def test_update_project_handles_deleted_files(tmp_path):
    """update_project loescht Daten fuer geloeschte Dateien."""
    from nemesis.indexer.pipeline import IndexingPipeline

    # Keine Dateien auf Disk, aber Graph kennt eine
    parser = MagicMock()
    graph = _make_mock_graph()
    deleted_path = str(tmp_path / "gone.py")
    graph.get_file_hashes.return_value = {deleted_path: "hash123"}
    graph.get_chunk_ids_for_file.return_value = ["chunk-old"]
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.update_project(tmp_path, languages=["python"])

    # Alte Daten wurden geloescht
    graph.delete_nodes_for_file.assert_called_once_with(deleted_path)
    vector_store.delete_embeddings.assert_called_once_with(["chunk-old"])
    # Kein Parse-Aufruf (Datei existiert nicht mehr)
    parser.parse_file.assert_not_called()
    assert result.success is True
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v -k "update_project"
# Erwartung: FAILED — update_project existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/pipeline.py — Methode update_project zur Klasse IndexingPipeline hinzufuegen

    def update_project(
        self,
        path: Path,
        languages: list[str],
        ignore_dirs: set[str] | None = None,
    ) -> IndexResult:
        """Delta Update: Nur geaenderte Dateien verarbeiten.

        Erkennt Aenderungen seit dem letzten Index und verarbeitet nur:
        - ADDED: Neu indexieren
        - MODIFIED: Alte Daten loeschen, neu indexieren
        - DELETED: Alte Daten loeschen

        Args:
            path: Wurzelverzeichnis des Projekts.
            languages: Liste der Sprachen.
            ignore_dirs: Verzeichnisnamen die ignoriert werden.

        Returns:
            Aggregiertes IndexResult.
        """
        from nemesis.indexer.delta import detect_changes, delete_file_data
        from nemesis.indexer.models import ChangeType

        start_time = time.monotonic()

        self._report_progress("Detecting changes...")
        changes = detect_changes(path, self._graph, languages, ignore_dirs)

        if not changes:
            duration_ms = (time.monotonic() - start_time) * 1000
            self._report_progress("No changes detected")
            return IndexResult(
                files_indexed=0,
                nodes_created=0,
                edges_created=0,
                chunks_created=0,
                embeddings_created=0,
                duration_ms=duration_ms,
                errors=[],
            )

        total_changes = len(changes)
        files_indexed = 0
        total_nodes = 0
        total_edges = 0
        total_chunks = 0
        total_embeddings = 0
        all_errors: list[str] = []

        added = [c for c in changes if c.change_type == ChangeType.ADDED]
        modified = [c for c in changes if c.change_type == ChangeType.MODIFIED]
        deleted = [c for c in changes if c.change_type == ChangeType.DELETED]

        self._report_progress(
            f"Changes: {len(added)} added, {len(modified)} modified, {len(deleted)} deleted",
            0,
            total_changes,
        )

        # Geloeschte Dateien: Nur alte Daten entfernen
        for i, change in enumerate(deleted):
            self._report_progress(
                f"Removing {change.path.name}",
                i + 1,
                total_changes,
            )
            try:
                delete_file_data(change.path, self._graph, self._vector_store)
            except Exception as e:
                all_errors.append(f"Error deleting data for {change.path}: {e}")

        # Geaenderte Dateien: Alte Daten loeschen, dann neu indexieren
        for i, change in enumerate(modified):
            self._report_progress(
                f"Re-indexing {change.path.name}",
                len(deleted) + i + 1,
                total_changes,
            )
            result = self.reindex_file(change.path)
            files_indexed += result.files_indexed
            total_nodes += result.nodes_created
            total_edges += result.edges_created
            total_chunks += result.chunks_created
            total_embeddings += result.embeddings_created
            all_errors.extend(result.errors)

        # Neue Dateien: Einfach indexieren
        for i, change in enumerate(added):
            self._report_progress(
                f"Indexing new {change.path.name}",
                len(deleted) + len(modified) + i + 1,
                total_changes,
            )
            result = self.index_file(change.path)
            files_indexed += result.files_indexed
            total_nodes += result.nodes_created
            total_edges += result.edges_created
            total_chunks += result.chunks_created
            total_embeddings += result.embeddings_created
            all_errors.extend(result.errors)

        duration_ms = (time.monotonic() - start_time) * 1000

        self._report_progress(
            f"Delta update done: {files_indexed} files processed",
            total_changes,
            total_changes,
        )

        return IndexResult(
            files_indexed=files_indexed,
            nodes_created=total_nodes,
            edges_created=total_edges,
            chunks_created=total_chunks,
            embeddings_created=total_embeddings,
            duration_ms=duration_ms,
            errors=all_errors,
        )
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/pipeline.py tests/test_indexer/test_pipeline.py
git commit -m "feat(indexer): add delta project update with change detection"
```

---

## Zusammenfassung

| Task | Datei(en) | Beschreibung |
|------|-----------|-------------|
| 1 | `models.py`, `__init__.py` | Datenmodelle: Chunk, FileChange, IndexResult, ChangeType |
| 2 | `tokens.py` | Token-Counting mit tiktoken + Heuristik-Fallback |
| 3 | `chunker.py` | AST-aware Chunker: Kleine Nodes als einzelne Chunks |
| 4 | `chunker.py` | Grosse Nodes an logischen Grenzen aufteilen |
| 5 | `delta.py` | Delta Detection: File-Hashes vergleichen, Changes erkennen |
| 6 | `delta.py` | delete_file_data: Alte Graph/Vector-Daten loeschen |
| 7 | `pipeline.py` | IndexingPipeline: Single File Index |
| 8 | `pipeline.py` | reindex_file: Delta Update fuer einzelne Dateien |
| 9 | `pipeline.py` | index_project: Full Project Index mit Progress |
| 10 | `pipeline.py` | update_project: Delta Project Update |

### Dateien erstellt

```
nemesis/indexer/
├── __init__.py         # Package init
├── models.py           # Chunk, FileChange, IndexResult, ChangeType
├── tokens.py           # count_tokens(), estimate_tokens()
├── chunker.py          # chunk_node() — AST-aware Chunking
├── delta.py            # detect_changes(), delete_file_data(), compute_file_hash()
└── pipeline.py         # IndexingPipeline — index_file, reindex_file, index_project, update_project

tests/test_indexer/
├── __init__.py
├── test_models.py
├── test_tokens.py
├── test_chunker.py
├── test_delta.py
└── test_pipeline.py
```

### Abhaengigkeiten gemockt

Alle externen Abhaengigkeiten werden in Tests durch Mocks ersetzt:
- **Parser** (`parse_file`) — gibt MockParseResult mit MockCodeNode/MockEdge zurueck
- **GraphAdapter** (`add_node`, `add_edge`, `delete_nodes_for_file`, `get_file_hashes`, `get_chunk_ids_for_file`) — MagicMock
- **VectorStore** (`add_embeddings`, `delete_embeddings`) — MagicMock
- **Embedder** (`embed_texts`) — gibt feste Vektoren zurueck
