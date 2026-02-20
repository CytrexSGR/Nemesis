"""Tests for nemesis.memory.models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from nemesis.memory.models import (
    AlternativeModel,
    ConventionModel,
    DecisionModel,
    RuleModel,
)


def test_rule_model_creation() -> None:
    rule = RuleModel(content="Use black for formatting", scope="project", source="manual")
    assert rule.id is not None
    assert isinstance(rule.created_at, datetime)
    assert rule.content == "Use black for formatting"
    assert rule.scope == "project"
    assert rule.source == "manual"


def test_rule_model_empty_content_rejected() -> None:
    with pytest.raises(ValidationError):
        RuleModel(content="")


def test_decision_model_creation() -> None:
    decision = DecisionModel(title="Use PostgreSQL", reasoning="Best fit", status="accepted")
    assert decision.id is not None
    assert decision.title == "Use PostgreSQL"
    assert decision.reasoning == "Best fit"
    assert decision.status == "accepted"


def test_decision_model_invalid_status() -> None:
    with pytest.raises(ValidationError):
        DecisionModel(title="Some decision", status="unknown")


def test_alternative_model_creation() -> None:
    alt = AlternativeModel(title="MySQL", reason_rejected="Lacks JSON support")
    assert alt.id is not None
    assert alt.title == "MySQL"
    assert alt.reason_rejected == "Lacks JSON support"


def test_convention_model_creation() -> None:
    conv = ConventionModel(pattern="snake_case", example="my_variable", scope="project")
    assert conv.id is not None
    assert conv.pattern == "snake_case"
    assert conv.example == "my_variable"
    assert conv.scope == "project"
    assert isinstance(conv.created_at, datetime)
