"""Tests for nemesis.memory.auto_learn — pattern detection and process_message."""

from __future__ import annotations

from unittest.mock import MagicMock

from nemesis.memory.auto_learn import detect_patterns, process_message

# ===================================================================
# Task 8: AutoLearn Pattern Detection
# ===================================================================


class TestGermanPatterns:
    """German-language pattern detection."""

    def test_ab_jetzt_immer(self) -> None:
        results = detect_patterns("Ab jetzt immer Type Hints verwenden")
        assert len(results) == 1
        intent = results[0]
        assert intent.intent_type == "rule"
        assert intent.confidence >= 0.8
        assert "Type Hints" in intent.content

    def test_nutze_nie(self) -> None:
        results = detect_patterns("Nutze nie print() für Logging")
        assert len(results) == 1
        intent = results[0]
        assert intent.intent_type == "rule"
        assert "print()" in intent.content

    def test_wir_haben_entschieden(self) -> None:
        results = detect_patterns("Wir haben entschieden, JWT zu nutzen")
        assert len(results) == 1
        intent = results[0]
        assert intent.intent_type == "decision"
        assert "JWT" in intent.content


class TestEnglishPatterns:
    """English-language pattern detection."""

    def test_always_use(self) -> None:
        results = detect_patterns("Always use parameterized queries")
        assert len(results) == 1
        intent = results[0]
        assert intent.intent_type == "rule"
        assert "parameterized queries" in intent.content

    def test_never_use(self) -> None:
        results = detect_patterns("Never use string concatenation for SQL")
        assert len(results) == 1
        intent = results[0]
        assert intent.intent_type == "rule"
        assert "string concatenation" in intent.content

    def test_we_decided(self) -> None:
        results = detect_patterns("We decided to use PostgreSQL over MySQL")
        assert len(results) == 1
        intent = results[0]
        assert intent.intent_type == "decision"
        assert "PostgreSQL" in intent.content


class TestEdgeCases:
    """Edge cases for pattern detection."""

    def test_no_pattern_detected(self) -> None:
        results = detect_patterns("This is just a normal comment about the code.")
        assert results == []

    def test_multiple_patterns_in_text(self) -> None:
        text = "Always use type hints. Never use Any type."
        results = detect_patterns(text)
        assert len(results) == 2
        assert all(r.intent_type == "rule" for r in results)


# ===================================================================
# Task 9: process_message Integration
# ===================================================================


class TestProcessMessage:
    """Tests for process_message graph integration."""

    def test_creates_rule_from_text(self) -> None:
        graph = MagicMock()
        results = process_message("Always use parameterized queries", graph)
        assert len(results) == 1
        assert results[0]["type"] == "rule"
        assert results[0]["model"] is not None
        graph.add_node.assert_called_once()

    def test_creates_decision_from_text(self) -> None:
        graph = MagicMock()
        results = process_message("We decided to use PostgreSQL over MySQL", graph)
        assert len(results) == 1
        assert results[0]["type"] == "decision"
        assert results[0]["model"] is not None
        graph.add_node.assert_called_once()

    def test_no_patterns_returns_empty(self) -> None:
        graph = MagicMock()
        results = process_message("Just a normal sentence with no patterns.", graph)
        assert results == []
        graph.add_node.assert_not_called()

    def test_german_rule_creates_node(self) -> None:
        graph = MagicMock()
        results = process_message("Nutze nie print() für Logging", graph)
        assert len(results) == 1
        assert results[0]["type"] == "rule"
        graph.add_node.assert_called_once()
        node = graph.add_node.call_args[0][0]
        assert node.node_type == "Rule"
        assert "print()" in node.properties["content"]

    def test_multiple_intents_create_multiple_nodes(self) -> None:
        graph = MagicMock()
        text = "Always use type hints. Never use Any type."
        results = process_message(text, graph)
        assert len(results) == 2
        assert graph.add_node.call_count == 2
        types = [r["type"] for r in results]
        assert types == ["rule", "rule"]
