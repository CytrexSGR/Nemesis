# Memory System — Pydantic Data Models

> **Arbeitspaket G1** — Teil 1 von 5 des Memory System Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pydantic Data Models für das Memory System (RuleModel, DecisionModel, AlternativeModel, ConventionModel).

**Tech Stack:** Python 3.11+, Pydantic, pytest
**Depends on:** [03-graph-layer](03-graph-layer.md), [06-mcp-server](06-mcp-server.md)

**Tasks in diesem Paket:** 1 (Task 1 von 10)

---

## Task 1: Pydantic Data Models

**Files:**
- `nemesis/memory/__init__.py`
- `nemesis/memory/models.py`
- `tests/test_memory/__init__.py`
- `tests/test_memory/test_models.py`

### Step 1 — Write failing test

```python
# tests/test_memory/__init__.py
```

```python
# tests/test_memory/test_models.py
"""Tests for memory Pydantic data models."""
import pytest
from pydantic import ValidationError
from datetime import datetime


def test_rule_model_creation():
    from nemesis.memory.models import RuleModel
    rule = RuleModel(content="Always use type hints", scope="project", source="manual")
    assert rule.content == "Always use type hints"
    assert rule.scope == "project"
    assert rule.source == "manual"
    assert rule.id is not None
    assert isinstance(rule.created_at, datetime)


def test_rule_model_empty_content_rejected():
    from nemesis.memory.models import RuleModel
    with pytest.raises(ValidationError):
        RuleModel(content="", scope="project", source="manual")


def test_decision_model_creation():
    from nemesis.memory.models import DecisionModel
    dec = DecisionModel(title="Use JWT", reasoning="Stateless auth", status="accepted")
    assert dec.title == "Use JWT"
    assert dec.status == "accepted"
    assert dec.id is not None


def test_decision_model_invalid_status():
    from nemesis.memory.models import DecisionModel
    with pytest.raises(ValidationError):
        DecisionModel(title="X", reasoning="Y", status="unknown")


def test_alternative_model_creation():
    from nemesis.memory.models import AlternativeModel
    alt = AlternativeModel(title="Session cookies", reason_rejected="Stateful, hard to scale")
    assert alt.title == "Session cookies"
    assert alt.reason_rejected == "Stateful, hard to scale"


def test_convention_model_creation():
    from nemesis.memory.models import ConventionModel
    conv = ConventionModel(pattern="snake_case", example="my_variable", scope="module")
    assert conv.pattern == "snake_case"
    assert conv.scope == "module"
    assert conv.id is not None
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_models.py -x -v 2>&1 | head -20
```

Tests fail because `nemesis/memory/models.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/__init__.py
"""Memory module — rules, decisions, conventions, auto-learn."""
```

```python
# nemesis/memory/models.py
"""Pydantic models for the Nemesis memory system."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid4())


class RuleModel(BaseModel):
    """A project rule stored in the graph."""
    id: str = Field(default_factory=_uuid)
    content: str = Field(min_length=1)
    scope: str = "project"
    source: str = "manual"
    created_at: datetime = Field(default_factory=_now)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class DecisionModel(BaseModel):
    """An architecture decision record."""
    id: str = Field(default_factory=_uuid)
    title: str = Field(min_length=1)
    reasoning: str = ""
    status: Literal["proposed", "accepted", "deprecated", "superseded"] = "proposed"
    created_at: datetime = Field(default_factory=_now)


class AlternativeModel(BaseModel):
    """A rejected alternative for a decision."""
    id: str = Field(default_factory=_uuid)
    title: str = Field(min_length=1)
    reason_rejected: str = ""


class ConventionModel(BaseModel):
    """A coding convention."""
    id: str = Field(default_factory=_uuid)
    pattern: str = Field(min_length=1)
    example: str = ""
    scope: str = "project"
    created_at: datetime = Field(default_factory=_now)
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_models.py -x -v
```

Expected: 6 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/__init__.py nemesis/memory/models.py \
        tests/test_memory/__init__.py tests/test_memory/test_models.py
git commit -m "feat(memory): add Pydantic data models for memory system

TDD Task 1/10 of 07-memory-system plan.
RuleModel, DecisionModel, AlternativeModel, ConventionModel with
validation and auto-generated IDs."
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | Pydantic data models | `models.py` | 6 |

---

**Navigation:**
- Vorheriges Paket: —
- Nächstes Paket: [07b-rules-manager.md](07b-rules-manager.md) (G2 — RulesManager)
