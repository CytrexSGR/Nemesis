"""Pydantic models for the Nemesis memory system."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid4())


class RuleModel(BaseModel):
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
    id: str = Field(default_factory=_uuid)
    title: str = Field(min_length=1)
    reasoning: str = ""
    status: Literal["proposed", "accepted", "deprecated", "superseded"] = "proposed"
    created_at: datetime = Field(default_factory=_now)


class AlternativeModel(BaseModel):
    id: str = Field(default_factory=_uuid)
    title: str = Field(min_length=1)
    reason_rejected: str = ""


class ConventionModel(BaseModel):
    id: str = Field(default_factory=_uuid)
    pattern: str = Field(min_length=1)
    example: str = ""
    scope: str = "project"
    created_at: datetime = Field(default_factory=_now)
