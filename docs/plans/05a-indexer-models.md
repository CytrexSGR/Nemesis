# 05a — Indexer Datenmodelle + Token-Counting

> **Arbeitspaket E1** — Teil 1 von 5 des Indexing Pipeline Plans

**Goal:** Definiere die zentralen Datenmodelle (Chunk, FileChange, IndexResult) und eine Token-Counting Utility fuer AST-aware Chunking.

**Tech Stack:** Python dataclasses, tiktoken (optional), Heuristik-Fallback

**Abhängigkeiten:** 02-rust-parser, 03-graph-layer, 04-vector-store

**Tasks in diesem Paket:** 1, 2

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

## Zusammenfassung E1

| Task | Datei(en) | Beschreibung |
|------|-----------|-------------|
| 1 | `models.py`, `__init__.py` | Datenmodelle: Chunk, FileChange, IndexResult, ChangeType |
| 2 | `tokens.py` | Token-Counting mit tiktoken + Heuristik-Fallback |

---

**Navigation:**
- Weiter: [05b — Chunking (E2)](05b-chunking.md)
