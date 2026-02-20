"""Memory module — rules, decisions, conventions, auto-learn."""

from nemesis.memory.auto_learn import MemoryIntent, detect_patterns, process_message
from nemesis.memory.context import SessionContext
from nemesis.memory.conventions import ConventionManager
from nemesis.memory.decisions import DecisionsManager
from nemesis.memory.models import (
    AlternativeModel,
    ConventionModel,
    DecisionModel,
    RuleModel,
)
from nemesis.memory.rules import RulesManager

__all__ = [
    "AlternativeModel",
    "ConventionManager",
    "ConventionModel",
    "DecisionModel",
    "DecisionsManager",
    "MemoryIntent",
    "RuleModel",
    "RulesManager",
    "SessionContext",
    "detect_patterns",
    "process_message",
]
