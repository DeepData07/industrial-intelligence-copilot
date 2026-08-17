"""Strict contracts for the bounded investigation planner."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.copilot.schemas import Intent


class PlannedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(min_length=3, max_length=180)


class KnowledgeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=3, max_length=300)
    failure_mode: Literal["TWF", "HDF", "PWF", "OSF", "RNF"] | None = None
    purpose: str = Field(min_length=3, max_length=180)


class InvestigationPlan(BaseModel):
    """An LLM proposal only; execution is controlled elsewhere."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: Intent = "incident_investigation"
    objective: str = Field(min_length=5, max_length=300)
    answerability: Literal["supported", "partially_supported", "unsupported"]
    evidence_needed: list[str] = Field(default_factory=list, max_length=6)
    tool_calls: list[PlannedToolCall] = Field(default_factory=list, max_length=4)
    knowledge_queries: list[KnowledgeQuery] = Field(default_factory=list, max_length=2)
    missing_data: list[str] = Field(default_factory=list, max_length=5)


class StructuredInvestigationContext(BaseModel):
    """Resolved conversation state passed to the planner instead of raw chat alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    cycle: int = Field(ge=0)
    incident_id: str | None = None
    previous_intent: str | None = None
    active_filters: dict[str, Any] = Field(default_factory=dict)
    active_variables: list[str] = Field(default_factory=list)
    last_comparison: str | None = None
    available_data: list[str] = Field(default_factory=list)
    unavailable_data: list[str] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    """One optional additional round; there is no unbounded loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enough_evidence: bool
    additional_tool_calls: list[PlannedToolCall] = Field(default_factory=list, max_length=2)
    additional_knowledge_queries: list[KnowledgeQuery] = Field(default_factory=list, max_length=2)
