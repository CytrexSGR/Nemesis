# Memory System — AutoLearn + Integration

> **Arbeitspaket G5** — Teil 5 von 5 des Memory System Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** AutoLearn Pattern Detection (DE + EN), process_message Integration und vollstaendiger Integrationstest mit Public API (Tasks 8, 9 und 10).

**Tech Stack:** Python 3.11+, Pydantic, Graph Adapter (from 03), pytest
**Depends on:** [03-graph-layer](03-graph-layer.md), [06-mcp-server](06-mcp-server.md), [07a-memory-models.md](07a-memory-models.md), [07b-rules-manager.md](07b-rules-manager.md), [07c-decisions-manager.md](07c-decisions-manager.md), [07d-conventions-context.md](07d-conventions-context.md)

**Tasks in diesem Paket:** 3 (Tasks 8–10 von 10)

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
| 8 | AutoLearn patterns | `auto_learn.py` | 8 |
| 9 | AutoLearn integration | `auto_learn.py` | 5 |
| 10 | Integration test | `test_integration.py` | 5 |

---

### Files created (gesamt ueber alle Pakete)

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

---

**Navigation:**
- Vorheriges Paket: [07d-conventions-context.md](07d-conventions-context.md) (G4 — ConventionManager + SessionContext)
- Nächstes Paket: —
