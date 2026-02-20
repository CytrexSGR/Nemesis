"""AutoLearn — automatic pattern detection for memory intents.

Detects rules, decisions, and conventions from natural-language text
(German and English) and optionally persists them into the knowledge graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.rules import RulesManager

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryIntent:
    """A detected memory intent from natural-language text.

    Attributes:
        intent_type: The type of intent ("rule", "decision", "convention").
        content: The extracted text content.
        confidence: Confidence score between 0.0 and 1.0.
    """

    intent_type: str
    content: str
    confidence: float


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each tuple: (compiled_regex, intent_type, confidence, group_idx)
_PATTERNS: list[tuple[re.Pattern[str], str, float, int]] = [
    # German patterns
    (re.compile(r"(?i)ab\s+jetzt\s+immer\s+(.+)", re.UNICODE), "rule", 0.85, 1),
    (re.compile(r"(?i)nutze?\s+nie(?:mals)?\s+(.+)", re.UNICODE), "rule", 0.85, 1),
    (re.compile(r"(?i)verwende?\s+nie(?:mals)?\s+(.+)", re.UNICODE), "rule", 0.85, 1),
    (
        re.compile(r"(?i)wir\s+haben\s+(?:uns\s+)?entschieden,?\s+(.+)", re.UNICODE),
        "decision",
        0.80,
        1,
    ),
    # English patterns
    (re.compile(r"(?i)always\s+use\s+(.+)", re.UNICODE), "rule", 0.85, 1),
    (re.compile(r"(?i)never\s+use\s+(.+)", re.UNICODE), "rule", 0.85, 1),
    (
        re.compile(r"(?i)we\s+decided\s+(?:to\s+)?(.+)", re.UNICODE),
        "decision",
        0.80,
        1,
    ),
]

# Sentence boundary pattern — splits on . ! ? followed by whitespace or end
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_patterns(text: str) -> list[MemoryIntent]:
    """Detect memory-relevant intents from natural-language text.

    Splits the input into sentences and matches each against known patterns
    for rules, decisions, and conventions in both German and English.

    Args:
        text: The input text to analyze.

    Returns:
        List of detected MemoryIntent instances.
    """
    if not text or not text.strip():
        return []

    sentences = _SENTENCE_SPLIT.split(text.strip())
    results: list[MemoryIntent] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        for pattern, intent_type, confidence, group_idx in _PATTERNS:
            match = pattern.search(sentence)
            if match:
                content = match.group(group_idx).strip().rstrip(".")
                results.append(
                    MemoryIntent(
                        intent_type=intent_type,
                        content=content,
                        confidence=confidence,
                    )
                )
                break  # one match per sentence

    return results


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------


def process_message(text: str, graph: object) -> list[dict]:
    """Detect intents and persist them as graph nodes.

    Args:
        text: The input text to analyze.
        graph: A GraphAdapter instance for persisting nodes.

    Returns:
        List of dicts with "type" and "model" keys for each created node.
    """
    intents = detect_patterns(text)
    if not intents:
        return []

    results: list[dict] = []
    rules_mgr = RulesManager(graph)  # type: ignore[arg-type]
    decisions_mgr = DecisionsManager(graph)  # type: ignore[arg-type]

    for intent in intents:
        if intent.intent_type == "rule":
            rule = rules_mgr.add_rule(intent.content, source="auto-learn")
            results.append({"type": "rule", "model": rule})
        elif intent.intent_type == "decision":
            dec = decisions_mgr.add_decision(intent.content, status="proposed")
            results.append({"type": "decision", "model": dec})

    return results
