"""Contracts for a compact curated domain knowledge corpus."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

KnowledgeAuthority = Literal["dataset_rule", "engineering_reference", "system_limit"]


class KnowledgeHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    section: str
    text: str = Field(min_length=1, max_length=1_800)
    authority: KnowledgeAuthority
    source: str
    failure_modes: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    score: float = Field(ge=0)
