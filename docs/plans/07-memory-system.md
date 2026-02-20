# Memory System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the persistent memory system for storing project rules, architecture decisions, conventions, and auto-learned patterns as graph nodes.

**Architecture:** Memory nodes (Rule, Decision, Alternative, Convention) are stored in the graph DB via the graph adapter from 03. Each module provides CRUD operations and links memory to code nodes via edges (APPLIES_TO, CHOSE, REJECTED, GOVERNS). Auto-learn detects patterns in German and English text.

**Tech Stack:** Python 3.11+, Pydantic, Graph Adapter (from 03), pytest
**Depends on:** [03-graph-layer](03-graph-layer.md), [06-mcp-server](06-mcp-server.md)

**Estimated Tasks:** 10

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

## Task 2: RulesManager Basics

**Files:**
- `nemesis/memory/rules.py`
- `tests/test_memory/test_rules.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_rules.py
"""Tests for RulesManager basic CRUD."""
import pytest
from unittest.mock import MagicMock
from nemesis.memory.rules import RulesManager
from nemesis.memory.models import RuleModel
from nemesis.graph.adapter import NodeData


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return RulesManager(mock_graph)


class TestAddRule:
    def test_add_rule_calls_add_node(self, manager, mock_graph):
        rule = manager.add_rule("Always use type hints", scope="project")
        assert isinstance(rule, RuleModel)
        mock_graph.add_node.assert_called_once()
        node = mock_graph.add_node.call_args[0][0]
        assert node.node_type == "Rule"
        assert node.properties["content"] == "Always use type hints"

    def test_add_rule_returns_model(self, manager):
        rule = manager.add_rule("Use pytest", source="auto")
        assert rule.content == "Use pytest"
        assert rule.source == "auto"


class TestGetRules:
    def test_get_rules_queries_graph(self, manager, mock_graph):
        mock_graph.query.return_value = [
            {"id": "r1", "content": "Rule A", "scope": "project",
             "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        ]
        rules = manager.get_rules()
        assert len(rules) == 1
        assert rules[0].content == "Rule A"

    def test_get_rules_empty(self, manager):
        rules = manager.get_rules()
        assert rules == []


class TestGetRuleById:
    def test_get_existing_rule(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="r1", node_type="Rule",
            properties={"content": "X", "scope": "project",
                        "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        )
        rule = manager.get_rule_by_id("r1")
        assert rule is not None
        assert rule.content == "X"

    def test_get_nonexistent_rule(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        assert manager.get_rule_by_id("nope") is None
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_rules.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/rules.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/rules.py
"""Rules manager — CRUD for project rules in the graph."""
from __future__ import annotations

from nemesis.graph.adapter import GraphAdapter, NodeData, EdgeData
from nemesis.memory.models import RuleModel


class RulesManager:
    """Manages Rule nodes in the graph."""

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    def add_rule(self, content: str, scope: str = "project",
                 source: str = "manual") -> RuleModel:
        rule = RuleModel(content=content, scope=scope, source=source)
        self._graph.add_node(NodeData(
            id=rule.id, node_type="Rule",
            properties={
                "content": rule.content, "scope": rule.scope,
                "created_at": rule.created_at.isoformat(), "source": rule.source,
            },
        ))
        return rule

    def get_rules(self) -> list[RuleModel]:
        rows = self._graph.query("MATCH (r:Rule) RETURN r.*")
        return [RuleModel(
            id=r["id"], content=r["content"], scope=r["scope"],
            created_at=r["created_at"], source=r["source"],
        ) for r in rows]

    def get_rule_by_id(self, rule_id: str) -> RuleModel | None:
        node = self._graph.get_node(rule_id)
        if node is None or node.node_type != "Rule":
            return None
        p = node.properties
        return RuleModel(
            id=node.id, content=p["content"], scope=p["scope"],
            created_at=p["created_at"], source=p["source"],
        )

    def update_rule(self, rule_id: str, **kwargs: object) -> RuleModel | None:
        existing = self.get_rule_by_id(rule_id)
        if existing is None:
            return None
        updated = existing.model_copy(update=kwargs)
        self._graph.add_node(NodeData(
            id=updated.id, node_type="Rule",
            properties={
                "content": updated.content, "scope": updated.scope,
                "created_at": updated.created_at.isoformat(), "source": updated.source,
            },
        ))
        return updated

    def delete_rule(self, rule_id: str) -> bool:
        if self.get_rule_by_id(rule_id) is None:
            return False
        self._graph.delete_node(rule_id)
        return True

    def get_rules_for_scope(self, scope: str) -> list[RuleModel]:
        rows = self._graph.query(
            "MATCH (r:Rule) WHERE r.scope = $scope RETURN r.*",
            parameters={"scope": scope},
        )
        return [RuleModel(
            id=r["id"], content=r["content"], scope=r["scope"],
            created_at=r["created_at"], source=r["source"],
        ) for r in rows]

    def link_rule_to_target(self, rule_id: str, target_id: str) -> None:
        self._graph.add_edge(EdgeData(
            source_id=rule_id, target_id=target_id, edge_type="APPLIES_TO",
        ))
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_rules.py -x -v
```

Expected: 5 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/rules.py tests/test_memory/test_rules.py
git commit -m "feat(memory): add RulesManager with basic CRUD operations

TDD Task 2/10 of 07-memory-system plan.
add_rule, get_rules, get_rule_by_id using graph adapter."
```

---

## Task 3: RulesManager Advanced

**Files:**
- `tests/test_memory/test_rules.py` (update)

### Step 1 — Write failing test

Append to `tests/test_memory/test_rules.py`:

```python
class TestUpdateRule:
    def test_update_existing_rule(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="r1", node_type="Rule",
            properties={"content": "Old", "scope": "project",
                        "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        )
        updated = manager.update_rule("r1", content="New content")
        assert updated is not None
        assert updated.content == "New content"
        assert mock_graph.add_node.call_count == 1

    def test_update_nonexistent_returns_none(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        assert manager.update_rule("nope", content="X") is None


class TestDeleteRule:
    def test_delete_existing_rule(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="r1", node_type="Rule",
            properties={"content": "X", "scope": "project",
                        "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        )
        assert manager.delete_rule("r1") is True
        mock_graph.delete_node.assert_called_once_with("r1")

    def test_delete_nonexistent_returns_false(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        assert manager.delete_rule("nope") is False


class TestRulesForScope:
    def test_get_rules_for_scope(self, manager, mock_graph):
        mock_graph.query.return_value = [
            {"id": "r1", "content": "A", "scope": "module",
             "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"},
        ]
        rules = manager.get_rules_for_scope("module")
        assert len(rules) == 1
        assert rules[0].scope == "module"
        mock_graph.query.assert_called_once()


class TestLinkRule:
    def test_link_rule_to_target(self, manager, mock_graph):
        manager.link_rule_to_target("r1", "file-001")
        mock_graph.add_edge.assert_called_once()
        edge = mock_graph.add_edge.call_args[0][0]
        assert edge.edge_type == "APPLIES_TO"
        assert edge.source_id == "r1"
        assert edge.target_id == "file-001"
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_rules.py -x -v 2>&1 | head -30
```

Tests should PASS immediately — implementation was included in Task 2.

### Step 3 — Implementation

Already implemented in Task 2 (`update_rule`, `delete_rule`, `get_rules_for_scope`, `link_rule_to_target`).

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_rules.py -x -v
```

Expected: 10 tests PASSED (5 from Task 2 + 5 new).

### Step 5 — Commit

```bash
git add tests/test_memory/test_rules.py
git commit -m "test(memory): add advanced RulesManager tests

TDD Task 3/10 of 07-memory-system plan.
Tests for update, delete, scope filtering, and APPLIES_TO edges."
```

---

## Task 4: DecisionsManager Basics

**Files:**
- `nemesis/memory/decisions.py`
- `tests/test_memory/test_decisions.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_decisions.py
"""Tests for DecisionsManager basic operations."""
import pytest
from unittest.mock import MagicMock
from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.models import DecisionModel, AlternativeModel
from nemesis.graph.adapter import NodeData


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return DecisionsManager(mock_graph)


class TestAddDecision:
    def test_add_decision(self, manager, mock_graph):
        dec = manager.add_decision("Use JWT", reasoning="Stateless")
        assert isinstance(dec, DecisionModel)
        assert dec.title == "Use JWT"
        mock_graph.add_node.assert_called_once()
        node = mock_graph.add_node.call_args[0][0]
        assert node.node_type == "Decision"

    def test_add_decision_default_status(self, manager):
        dec = manager.add_decision("X")
        assert dec.status == "proposed"


class TestGetDecisions:
    def test_get_decisions(self, manager, mock_graph):
        mock_graph.query.return_value = [
            {"id": "d1", "title": "JWT", "reasoning": "Stateless",
             "created_at": "2026-01-01T00:00:00+00:00", "status": "accepted"},
        ]
        decs = manager.get_decisions()
        assert len(decs) == 1
        assert decs[0].title == "JWT"


class TestAddAlternative:
    def test_add_alternative(self, manager, mock_graph):
        alt = manager.add_alternative("d1", "Sessions", "Stateful")
        assert isinstance(alt, AlternativeModel)
        assert alt.title == "Sessions"
        # Should create node + REJECTED edge
        assert mock_graph.add_node.call_count == 1
        assert mock_graph.add_edge.call_count == 1
        edge = mock_graph.add_edge.call_args[0][0]
        assert edge.edge_type == "REJECTED"
        assert edge.source_id == "d1"

    def test_add_alternative_chose_edge(self, manager, mock_graph):
        manager.add_chose_link("d1", "conv-1")
        mock_graph.add_edge.assert_called_once()
        edge = mock_graph.add_edge.call_args[0][0]
        assert edge.edge_type == "CHOSE"
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_decisions.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/decisions.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/decisions.py
"""Decisions manager — CRUD for architecture decisions in the graph."""
from __future__ import annotations

from typing import Literal

from nemesis.graph.adapter import GraphAdapter, NodeData, EdgeData
from nemesis.memory.models import DecisionModel, AlternativeModel


class DecisionsManager:
    """Manages Decision and Alternative nodes in the graph."""

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    def add_decision(self, title: str, reasoning: str = "",
                     status: str = "proposed") -> DecisionModel:
        dec = DecisionModel(title=title, reasoning=reasoning, status=status)
        self._graph.add_node(NodeData(
            id=dec.id, node_type="Decision",
            properties={
                "title": dec.title, "reasoning": dec.reasoning,
                "created_at": dec.created_at.isoformat(), "status": dec.status,
            },
        ))
        return dec

    def get_decisions(self) -> list[DecisionModel]:
        rows = self._graph.query("MATCH (d:Decision) RETURN d.*")
        return [DecisionModel(
            id=r["id"], title=r["title"], reasoning=r["reasoning"],
            created_at=r["created_at"], status=r["status"],
        ) for r in rows]

    def add_alternative(self, decision_id: str, title: str,
                        reason_rejected: str = "") -> AlternativeModel:
        alt = AlternativeModel(title=title, reason_rejected=reason_rejected)
        self._graph.add_node(NodeData(
            id=alt.id, node_type="Alternative",
            properties={"title": alt.title, "reason_rejected": alt.reason_rejected},
        ))
        self._graph.add_edge(EdgeData(
            source_id=decision_id, target_id=alt.id, edge_type="REJECTED",
        ))
        return alt

    def add_chose_link(self, decision_id: str, target_id: str) -> None:
        self._graph.add_edge(EdgeData(
            source_id=decision_id, target_id=target_id, edge_type="CHOSE",
        ))

    def update_decision_status(self, decision_id: str,
                               status: str) -> DecisionModel | None:
        node = self._graph.get_node(decision_id)
        if node is None or node.node_type != "Decision":
            return None
        p = node.properties
        dec = DecisionModel(
            id=node.id, title=p["title"], reasoning=p["reasoning"],
            created_at=p["created_at"], status=status,
        )
        self._graph.add_node(NodeData(
            id=dec.id, node_type="Decision",
            properties={
                "title": dec.title, "reasoning": dec.reasoning,
                "created_at": dec.created_at.isoformat(), "status": dec.status,
            },
        ))
        return dec

    def get_decision_with_alternatives(self, decision_id: str
                                       ) -> tuple[DecisionModel | None, list[AlternativeModel]]:
        node = self._graph.get_node(decision_id)
        if node is None or node.node_type != "Decision":
            return None, []
        p = node.properties
        dec = DecisionModel(
            id=node.id, title=p["title"], reasoning=p["reasoning"],
            created_at=p["created_at"], status=p["status"],
        )
        neighbors = self._graph.get_neighbors(decision_id, edge_type="REJECTED", direction="outgoing")
        alts = [AlternativeModel(
            id=n.id, title=n.properties["title"],
            reason_rejected=n.properties.get("reason_rejected", ""),
        ) for n in neighbors if n.node_type == "Alternative"]
        return dec, alts

    def delete_decision(self, decision_id: str) -> bool:
        node = self._graph.get_node(decision_id)
        if node is None or node.node_type != "Decision":
            return False
        neighbors = self._graph.get_neighbors(decision_id, edge_type="REJECTED", direction="outgoing")
        for n in neighbors:
            self._graph.delete_node(n.id)
        self._graph.delete_node(decision_id)
        return True
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_decisions.py -x -v
```

Expected: 5 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/decisions.py tests/test_memory/test_decisions.py
git commit -m "feat(memory): add DecisionsManager with CRUD and alternative linking

TDD Task 4/10 of 07-memory-system plan.
add_decision, get_decisions, add_alternative, add_chose_link
with REJECTED and CHOSE edges."
```

---

## Task 5: DecisionsManager Advanced

**Files:**
- `tests/test_memory/test_decisions.py` (update)

### Step 1 — Write failing test

Append to `tests/test_memory/test_decisions.py`:

```python
class TestUpdateDecisionStatus:
    def test_update_status(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="d1", node_type="Decision",
            properties={"title": "JWT", "reasoning": "Stateless",
                        "created_at": "2026-01-01T00:00:00+00:00", "status": "proposed"},
        )
        dec = manager.update_decision_status("d1", "accepted")
        assert dec is not None
        assert dec.status == "accepted"

    def test_update_nonexistent_returns_none(self, manager, mock_graph):
        mock_graph.get_node.return_value = None
        assert manager.update_decision_status("nope", "accepted") is None


class TestGetDecisionWithAlternatives:
    def test_get_with_alternatives(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="d1", node_type="Decision",
            properties={"title": "JWT", "reasoning": "R",
                        "created_at": "2026-01-01T00:00:00+00:00", "status": "accepted"},
        )
        mock_graph.get_neighbors.return_value = [
            NodeData(id="a1", node_type="Alternative",
                     properties={"title": "Sessions", "reason_rejected": "Stateful"}),
        ]
        dec, alts = manager.get_decision_with_alternatives("d1")
        assert dec is not None
        assert len(alts) == 1
        assert alts[0].title == "Sessions"


class TestDeleteDecision:
    def test_delete_with_alternatives(self, manager, mock_graph):
        mock_graph.get_node.return_value = NodeData(
            id="d1", node_type="Decision",
            properties={"title": "JWT", "reasoning": "R",
                        "created_at": "2026-01-01T00:00:00+00:00", "status": "accepted"},
        )
        mock_graph.get_neighbors.return_value = [
            NodeData(id="a1", node_type="Alternative",
                     properties={"title": "X", "reason_rejected": "Y"}),
        ]
        assert manager.delete_decision("d1") is True
        assert mock_graph.delete_node.call_count == 2  # alt + decision
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_decisions.py -x -v 2>&1 | head -30
```

Should PASS — implementation included in Task 4.

### Step 3 — Implementation

Already implemented in Task 4.

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_decisions.py -x -v
```

Expected: 9 tests PASSED (5 from Task 4 + 4 new).

### Step 5 — Commit

```bash
git add tests/test_memory/test_decisions.py
git commit -m "test(memory): add advanced DecisionsManager tests

TDD Task 5/10 of 07-memory-system plan.
Tests for status updates, decision+alternatives retrieval, cascade delete."
```

---

## Task 6: ConventionManager

**Files:**
- `nemesis/memory/conventions.py`
- `tests/test_memory/test_conventions.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_conventions.py
"""Tests for ConventionManager."""
import pytest
from unittest.mock import MagicMock
from nemesis.memory.conventions import ConventionManager
from nemesis.memory.models import ConventionModel
from nemesis.graph.adapter import NodeData


@pytest.fixture
def mock_graph():
    adapter = MagicMock()
    adapter.get_node.return_value = None
    adapter.query.return_value = []
    return adapter


@pytest.fixture
def manager(mock_graph):
    return ConventionManager(mock_graph)


def test_add_convention(manager, mock_graph):
    conv = manager.add_convention("snake_case", example="my_var", scope="project")
    assert isinstance(conv, ConventionModel)
    assert conv.pattern == "snake_case"
    mock_graph.add_node.assert_called_once()
    node = mock_graph.add_node.call_args[0][0]
    assert node.node_type == "Convention"


def test_get_conventions(manager, mock_graph):
    mock_graph.query.return_value = [
        {"id": "c1", "pattern": "snake_case", "example": "my_var",
         "scope": "project", "created_at": "2026-01-01T00:00:00+00:00"},
    ]
    convs = manager.get_conventions()
    assert len(convs) == 1
    assert convs[0].pattern == "snake_case"


def test_link_convention_governs(manager, mock_graph):
    manager.link_governs("c1", "module-001")
    mock_graph.add_edge.assert_called_once()
    edge = mock_graph.add_edge.call_args[0][0]
    assert edge.edge_type == "GOVERNS"
    assert edge.source_id == "c1"
    assert edge.target_id == "module-001"


def test_delete_convention(manager, mock_graph):
    mock_graph.get_node.return_value = NodeData(
        id="c1", node_type="Convention",
        properties={"pattern": "X", "example": "", "scope": "project",
                    "created_at": "2026-01-01T00:00:00+00:00"},
    )
    assert manager.delete_convention("c1") is True
    mock_graph.delete_node.assert_called_once_with("c1")
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_conventions.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/conventions.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/conventions.py
"""Convention manager — CRUD for coding conventions in the graph."""
from __future__ import annotations

from nemesis.graph.adapter import GraphAdapter, NodeData, EdgeData
from nemesis.memory.models import ConventionModel


class ConventionManager:
    """Manages Convention nodes in the graph."""

    def __init__(self, graph: GraphAdapter) -> None:
        self._graph = graph

    def add_convention(self, pattern: str, example: str = "",
                       scope: str = "project") -> ConventionModel:
        conv = ConventionModel(pattern=pattern, example=example, scope=scope)
        self._graph.add_node(NodeData(
            id=conv.id, node_type="Convention",
            properties={
                "pattern": conv.pattern, "example": conv.example,
                "scope": conv.scope, "created_at": conv.created_at.isoformat(),
            },
        ))
        return conv

    def get_conventions(self) -> list[ConventionModel]:
        rows = self._graph.query("MATCH (c:Convention) RETURN c.*")
        return [ConventionModel(
            id=r["id"], pattern=r["pattern"], example=r["example"],
            scope=r["scope"], created_at=r["created_at"],
        ) for r in rows]

    def link_governs(self, convention_id: str, target_id: str) -> None:
        self._graph.add_edge(EdgeData(
            source_id=convention_id, target_id=target_id, edge_type="GOVERNS",
        ))

    def delete_convention(self, convention_id: str) -> bool:
        node = self._graph.get_node(convention_id)
        if node is None or node.node_type != "Convention":
            return False
        self._graph.delete_node(convention_id)
        return True
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_conventions.py -x -v
```

Expected: 4 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/conventions.py tests/test_memory/test_conventions.py
git commit -m "feat(memory): add ConventionManager with GOVERNS edge support

TDD Task 6/10 of 07-memory-system plan.
add_convention, get_conventions, link_governs, delete_convention."
```

---

## Task 7: SessionContext

**Files:**
- `nemesis/memory/context.py`
- `tests/test_memory/test_context.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_context.py
"""Tests for SessionContext — in-memory session tracking."""
import pytest
from nemesis.memory.context import SessionContext


def test_empty_session():
    ctx = SessionContext()
    assert ctx.get_queries() == []
    assert ctx.get_results() == []


def test_add_query():
    ctx = SessionContext()
    ctx.add_query("What does auth do?")
    ctx.add_query("Show dependencies")
    assert len(ctx.get_queries()) == 2
    assert ctx.get_queries()[0] == "What does auth do?"


def test_add_result():
    ctx = SessionContext()
    ctx.add_result("query1", {"files": ["auth.py"], "symbols": ["login"]})
    results = ctx.get_results()
    assert len(results) == 1
    assert results[0]["query"] == "query1"
    assert results[0]["data"]["files"] == ["auth.py"]


def test_build_summary():
    ctx = SessionContext()
    ctx.add_query("auth flow")
    ctx.add_result("auth flow", {"files": ["auth.py"]})
    summary = ctx.build_summary()
    assert "auth flow" in summary
    assert "auth.py" in summary


def test_clear():
    ctx = SessionContext()
    ctx.add_query("test")
    ctx.add_result("test", {"x": 1})
    ctx.clear()
    assert ctx.get_queries() == []
    assert ctx.get_results() == []
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_context.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/context.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/context.py
"""Session context — tracks queries and results within a session."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionContext:
    """In-memory session context (not persisted to graph)."""

    _queries: list[str] = field(default_factory=list)
    _results: list[dict] = field(default_factory=list)

    def add_query(self, query: str) -> None:
        self._queries.append(query)

    def add_result(self, query: str, data: dict) -> None:
        self._results.append({"query": query, "data": data})

    def get_queries(self) -> list[str]:
        return list(self._queries)

    def get_results(self) -> list[dict]:
        return list(self._results)

    def build_summary(self) -> str:
        lines: list[str] = []
        for entry in self._results:
            q = entry["query"]
            data = entry["data"]
            files = data.get("files", [])
            symbols = data.get("symbols", [])
            parts = [f"Query: {q}"]
            if files:
                parts.append(f"  Files: {', '.join(files)}")
            if symbols:
                parts.append(f"  Symbols: {', '.join(symbols)}")
            lines.extend(parts)
        if not lines:
            for q in self._queries:
                lines.append(f"Query: {q}")
        return "\n".join(lines) if lines else ""

    def clear(self) -> None:
        self._queries.clear()
        self._results.clear()
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_context.py -x -v
```

Expected: 5 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/context.py tests/test_memory/test_context.py
git commit -m "feat(memory): add SessionContext for in-memory session tracking

TDD Task 7/10 of 07-memory-system plan.
Tracks queries, results, builds text summary. Not persisted to graph."
```

---

## Task 8: AutoLearn Pattern Detection

**Files:**
- `nemesis/memory/auto_learn.py`
- `tests/test_memory/test_auto_learn.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_auto_learn.py
"""Tests for auto-learn pattern detection."""
import pytest
from nemesis.memory.auto_learn import detect_patterns, MemoryIntent


class TestGermanPatterns:
    def test_ab_jetzt_immer(self):
        intents = detect_patterns("Ab jetzt immer Type Hints verwenden")
        assert len(intents) == 1
        assert intents[0].intent_type == "rule"
        assert "Type Hints verwenden" in intents[0].content
        assert intents[0].confidence >= 0.8

    def test_nutze_nie(self):
        intents = detect_patterns("Nutze nie print() für Logging")
        assert len(intents) == 1
        assert intents[0].intent_type == "rule"
        assert "print() für Logging" in intents[0].content

    def test_wir_haben_entschieden(self):
        intents = detect_patterns("Wir haben entschieden, JWT zu nutzen")
        assert len(intents) == 1
        assert intents[0].intent_type == "decision"
        assert "JWT zu nutzen" in intents[0].content


class TestEnglishPatterns:
    def test_always_use(self):
        intents = detect_patterns("Always use parameterized queries")
        assert len(intents) == 1
        assert intents[0].intent_type == "rule"
        assert "parameterized queries" in intents[0].content

    def test_never_use(self):
        intents = detect_patterns("Never use string concatenation for SQL")
        assert len(intents) == 1
        assert intents[0].intent_type == "rule"

    def test_we_decided(self):
        intents = detect_patterns("We decided to use PostgreSQL over MySQL")
        assert len(intents) == 1
        assert intents[0].intent_type == "decision"


class TestEdgeCases:
    def test_no_pattern_detected(self):
        intents = detect_patterns("Just a normal sentence about code")
        assert intents == []

    def test_multiple_patterns_in_text(self):
        text = "Always use type hints. Never use Any type."
        intents = detect_patterns(text)
        assert len(intents) == 2
        assert all(i.intent_type == "rule" for i in intents)
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_auto_learn.py -x -v 2>&1 | head -20
```

Fails because `nemesis/memory/auto_learn.py` does not exist.

### Step 3 — Implement

```python
# nemesis/memory/auto_learn.py
"""Auto-learn — detect memory intents from natural language (DE + EN)."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryIntent:
    """A detected memory intent from text."""
    intent_type: str  # "rule" or "decision"
    content: str
    confidence: float


# (compiled_regex, intent_type, confidence, content_group_index)
_PATTERNS: list[tuple[re.Pattern[str], str, float, int]] = [
    # German rules
    (re.compile(r"[Aa]b jetzt (?:immer|grundsätzlich)\s+(.+)", re.IGNORECASE), "rule", 0.9, 1),
    (re.compile(r"[Nn]utze nie(?:mals)?\s+(.+)", re.IGNORECASE), "rule", 0.9, 1),
    (re.compile(r"[Vv]erwende (?:immer|nie(?:mals)?)\s+(.+)", re.IGNORECASE), "rule", 0.85, 1),
    (re.compile(r"[Bb]enutze (?:immer|nie(?:mals)?)\s+(.+)", re.IGNORECASE), "rule", 0.85, 1),
    # German decisions
    (re.compile(r"[Ww]ir haben entschieden,?\s+(.+)", re.IGNORECASE), "decision", 0.9, 1),
    (re.compile(r"[Ee]ntscheidung:\s+(.+)", re.IGNORECASE), "decision", 0.85, 1),
    # English rules
    (re.compile(r"[Aa]lways use\s+(.+)"), "rule", 0.9, 1),
    (re.compile(r"[Nn]ever use\s+(.+)"), "rule", 0.9, 1),
    (re.compile(r"[Aa]lways prefer\s+(.+)"), "rule", 0.85, 1),
    (re.compile(r"[Nn]ever commit\s+(.+)"), "rule", 0.85, 1),
    # English decisions
    (re.compile(r"[Ww]e decided (?:to )?\s*(.+)"), "decision", 0.9, 1),
    (re.compile(r"[Dd]ecision:\s+(.+)"), "decision", 0.85, 1),
]


def detect_patterns(text: str) -> list[MemoryIntent]:
    """Detect memory intents from free-form text."""
    intents: list[MemoryIntent] = []
    # Split on sentence boundaries for multi-pattern detection
    sentences = re.split(r"[.!?\n]+", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for pattern, intent_type, confidence, group_idx in _PATTERNS:
            m = pattern.search(sentence)
            if m:
                content = m.group(group_idx).strip().rstrip(".")
                intents.append(MemoryIntent(
                    intent_type=intent_type,
                    content=content,
                    confidence=confidence,
                ))
                break  # one match per sentence
    return intents
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_auto_learn.py -x -v
```

Expected: 8 tests PASSED.

### Step 5 — Commit

```bash
git add nemesis/memory/auto_learn.py tests/test_memory/test_auto_learn.py
git commit -m "feat(memory): add auto-learn pattern detection for DE and EN

TDD Task 8/10 of 07-memory-system plan.
Regex-based detection of rule and decision intents from natural text.
Supports German (ab jetzt immer, nutze nie, wir haben entschieden)
and English (always use, never use, we decided)."
```

---

## Task 9: AutoLearn Integration

**Files:**
- `nemesis/memory/auto_learn.py` (update)
- `tests/test_memory/test_auto_learn.py` (update)

### Step 1 — Write failing test

Append to `tests/test_memory/test_auto_learn.py`:

```python
from unittest.mock import MagicMock
from nemesis.memory.auto_learn import process_message


class TestProcessMessage:
    @pytest.fixture
    def mock_graph(self):
        adapter = MagicMock()
        adapter.get_node.return_value = None
        adapter.query.return_value = []
        return adapter

    def test_creates_rule_from_text(self, mock_graph):
        results = process_message("Always use type hints", mock_graph)
        assert len(results) == 1
        assert results[0]["type"] == "rule"
        assert mock_graph.add_node.call_count == 1

    def test_creates_decision_from_text(self, mock_graph):
        results = process_message("We decided to use FastAPI", mock_graph)
        assert len(results) == 1
        assert results[0]["type"] == "decision"
        assert mock_graph.add_node.call_count == 1

    def test_no_patterns_returns_empty(self, mock_graph):
        results = process_message("Just talking about code", mock_graph)
        assert results == []
        mock_graph.add_node.assert_not_called()

    def test_german_rule_creates_node(self, mock_graph):
        results = process_message("Ab jetzt immer Docstrings schreiben", mock_graph)
        assert len(results) == 1
        assert results[0]["type"] == "rule"

    def test_multiple_intents_create_multiple_nodes(self, mock_graph):
        text = "Always use pytest. We decided to drop unittest."
        results = process_message(text, mock_graph)
        assert len(results) == 2
        assert mock_graph.add_node.call_count == 2
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_auto_learn.py::TestProcessMessage -x -v 2>&1 | head -20
```

Fails because `process_message` does not exist yet.

### Step 3 — Implement

Add to `nemesis/memory/auto_learn.py`:

```python
from nemesis.memory.rules import RulesManager
from nemesis.memory.decisions import DecisionsManager


def process_message(text: str, graph: GraphAdapter) -> list[dict]:
    """Detect patterns in text and create memory nodes.

    Returns list of dicts with 'type' and 'model' keys.
    """
    intents = detect_patterns(text)
    if not intents:
        return []

    results: list[dict] = []
    rules_mgr = RulesManager(graph)
    decisions_mgr = DecisionsManager(graph)

    for intent in intents:
        if intent.intent_type == "rule":
            rule = rules_mgr.add_rule(intent.content, source="auto-learn")
            results.append({"type": "rule", "model": rule})
        elif intent.intent_type == "decision":
            dec = decisions_mgr.add_decision(intent.content, status="proposed")
            results.append({"type": "decision", "model": dec})

    return results
```

Also add the import at the top of the file:

```python
from nemesis.graph.adapter import GraphAdapter
```

### Step 4 — Run tests, verify they PASS

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_auto_learn.py -x -v
```

Expected: 13 tests PASSED (8 from Task 8 + 5 new).

### Step 5 — Commit

```bash
git add nemesis/memory/auto_learn.py tests/test_memory/test_auto_learn.py
git commit -m "feat(memory): add process_message to auto-create memory nodes

TDD Task 9/10 of 07-memory-system plan.
process_message detects intents and creates Rule/Decision nodes
via RulesManager and DecisionsManager."
```

---

## Task 10: Integration Test

**Files:**
- `nemesis/memory/__init__.py` (update — export public API)
- `tests/test_memory/test_integration.py`

### Step 1 — Write failing test

```python
# tests/test_memory/test_integration.py
"""Integration tests for the full memory system."""
import pytest
from unittest.mock import MagicMock, call
from nemesis.graph.adapter import NodeData, EdgeData
from nemesis.memory.rules import RulesManager
from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.conventions import ConventionManager
from nemesis.memory.context import SessionContext
from nemesis.memory.auto_learn import detect_patterns, process_message


class FakeGraph:
    """In-memory fake graph for integration tests."""

    def __init__(self):
        self._nodes: dict[str, NodeData] = {}
        self._edges: list[EdgeData] = []

    def add_node(self, node: NodeData) -> None:
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> NodeData | None:
        return self._nodes.get(node_id)

    def delete_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._edges = [e for e in self._edges
                       if e.source_id != node_id and e.target_id != node_id]

    def add_edge(self, edge: EdgeData) -> None:
        self._edges.append(edge)

    def query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        # Simple fake: return all nodes of the queried type
        if ":Rule" in cypher:
            return [{"id": n.id, **n.properties}
                    for n in self._nodes.values() if n.node_type == "Rule"]
        if ":Decision" in cypher:
            return [{"id": n.id, **n.properties}
                    for n in self._nodes.values() if n.node_type == "Decision"]
        if ":Convention" in cypher:
            return [{"id": n.id, **n.properties}
                    for n in self._nodes.values() if n.node_type == "Convention"]
        return []

    def get_neighbors(self, node_id: str, edge_type: str | None = None,
                      direction: str = "outgoing") -> list[NodeData]:
        result = []
        for e in self._edges:
            if edge_type and e.edge_type != edge_type:
                continue
            if direction in ("outgoing", "both") and e.source_id == node_id:
                n = self._nodes.get(e.target_id)
                if n:
                    result.append(n)
            if direction in ("incoming", "both") and e.target_id == node_id:
                n = self._nodes.get(e.source_id)
                if n:
                    result.append(n)
        return result


@pytest.fixture
def graph():
    return FakeGraph()


def test_full_rule_lifecycle(graph):
    mgr = RulesManager(graph)
    rule = mgr.add_rule("Always use type hints", scope="project")
    assert mgr.get_rules()[0].content == "Always use type hints"
    updated = mgr.update_rule(rule.id, content="Use strict type hints")
    assert updated.content == "Use strict type hints"
    assert mgr.delete_rule(rule.id) is True
    assert mgr.get_rules() == []


def test_decision_with_alternatives(graph):
    mgr = DecisionsManager(graph)
    dec = mgr.add_decision("Use PostgreSQL", reasoning="ACID compliance")
    mgr.add_alternative(dec.id, "MongoDB", reason_rejected="No ACID")
    mgr.add_alternative(dec.id, "MySQL", reason_rejected="License concerns")
    found_dec, alts = mgr.get_decision_with_alternatives(dec.id)
    assert found_dec.title == "Use PostgreSQL"
    assert len(alts) == 2


def test_convention_with_governs_edge(graph):
    conv_mgr = ConventionManager(graph)
    conv = conv_mgr.add_convention("snake_case", example="my_func", scope="project")
    # Simulate a module node
    graph.add_node(NodeData(id="mod-1", node_type="Module",
                            properties={"name": "auth", "path": "src/auth"}))
    conv_mgr.link_governs(conv.id, "mod-1")
    # Verify edge exists
    neighbors = graph.get_neighbors(conv.id, edge_type="GOVERNS")
    assert len(neighbors) == 1
    assert neighbors[0].id == "mod-1"


def test_auto_learn_creates_graph_nodes(graph):
    results = process_message("Always use dataclasses. We decided to use Pydantic.", graph)
    assert len(results) == 2
    rules = [r for r in results if r["type"] == "rule"]
    decisions = [r for r in results if r["type"] == "decision"]
    assert len(rules) == 1
    assert len(decisions) == 1
    # Verify nodes exist in graph
    rule_node = graph.get_node(rules[0]["model"].id)
    assert rule_node is not None
    assert rule_node.node_type == "Rule"


def test_session_context_with_memory(graph):
    ctx = SessionContext()
    mgr = RulesManager(graph)
    rule = mgr.add_rule("Use pytest")
    ctx.add_query("What are our testing rules?")
    ctx.add_result("What are our testing rules?",
                   {"files": [], "symbols": [], "rules": [rule.content]})
    summary = ctx.build_summary()
    assert "testing rules" in summary
```

### Step 2 — Run tests, verify they FAIL

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/test_integration.py -x -v 2>&1 | head -20
```

Should PASS if all previous tasks are implemented.

### Step 3 — Update module init

```python
# nemesis/memory/__init__.py
"""Memory module — rules, decisions, conventions, auto-learn."""
from nemesis.memory.models import RuleModel, DecisionModel, AlternativeModel, ConventionModel
from nemesis.memory.rules import RulesManager
from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.conventions import ConventionManager
from nemesis.memory.context import SessionContext
from nemesis.memory.auto_learn import detect_patterns, process_message, MemoryIntent

__all__ = [
    "RuleModel", "DecisionModel", "AlternativeModel", "ConventionModel",
    "RulesManager", "DecisionsManager", "ConventionManager",
    "SessionContext",
    "detect_patterns", "process_message", "MemoryIntent",
]
```

### Step 4 — Run all memory tests

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_memory/ -x -v
```

Expected: ~52 tests PASSED across all test files.

### Step 5 — Commit

```bash
git add nemesis/memory/__init__.py tests/test_memory/test_integration.py
git commit -m "feat(memory): add integration tests and public module API

TDD Task 10/10 of 07-memory-system plan.
Full lifecycle tests with FakeGraph adapter. Exports all public
classes and functions from nemesis.memory."
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | Pydantic data models | `models.py` | 6 |
| 2 | RulesManager basics | `rules.py` | 5 |
| 3 | RulesManager advanced | `test_rules.py` | 5 |
| 4 | DecisionsManager basics | `decisions.py` | 5 |
| 5 | DecisionsManager advanced | `test_decisions.py` | 4 |
| 6 | ConventionManager | `conventions.py` | 4 |
| 7 | SessionContext | `context.py` | 5 |
| 8 | AutoLearn patterns | `auto_learn.py` | 8 |
| 9 | AutoLearn integration | `auto_learn.py` | 5 |
| 10 | Integration test | `test_integration.py` | 5 |
| **Total** | | **6 source + 6 test files** | **~52 tests** |

### Files created

```
nemesis/memory/
├── __init__.py       # Public API exports
├── models.py         # RuleModel, DecisionModel, AlternativeModel, ConventionModel
├── rules.py          # RulesManager — CRUD + APPLIES_TO edges
├── decisions.py      # DecisionsManager — CRUD + CHOSE/REJECTED edges
├── conventions.py    # ConventionManager — CRUD + GOVERNS edges
├── context.py        # SessionContext — in-memory session tracking
└── auto_learn.py     # Pattern detection + process_message integration

tests/test_memory/
├── __init__.py
├── test_models.py         # Pydantic model validation
├── test_rules.py          # RulesManager unit tests
├── test_decisions.py      # DecisionsManager unit tests
├── test_conventions.py    # ConventionManager unit tests
├── test_context.py        # SessionContext unit tests
├── test_auto_learn.py     # Pattern detection + integration
└── test_integration.py    # Full lifecycle with FakeGraph
```

### Graph Schema (Memory Nodes)

```
:Rule          {id, content, scope, created_at, source}
:Decision      {id, title, reasoning, created_at, status}
:Alternative   {id, title, reason_rejected}
:Convention    {id, pattern, example, scope, created_at}

(:Rule)-[:APPLIES_TO]->(:File|:Module|:Class|:Project)
(:Decision)-[:CHOSE]->(:Convention|:Class|:Module)
(:Decision)-[:REJECTED]->(:Alternative)
(:Convention)-[:GOVERNS]->(:Module|:File)
```

### Dependencies

```toml
[project]
dependencies = [
    "pydantic>=2.0",
]
```
