"""Validated plan and evidence contracts for the controlled copilot layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.analytics.models import AnalysisFilters

Intent = Literal[
    "dataset_summary",
    "failure_rate",
    "failed_healthy_comparison",
    "product_comparison",
    "failure_relationship",
    "failure_investigation",
    "observation_lookup",
    "similar_conditions",
    "data_quality",
    "model_risk",
    "incident_investigation",
    "unavailable_data",
    "unsupported_question",
]


class ToolCall(BaseModel):
    """A proposed call to one whitelisted deterministic tool."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    """Validated intent plus a bounded list of tool calls; no executable code is allowed."""

    model_config = ConfigDict(frozen=True)

    intent: Intent
    tools: list[ToolCall] = Field(default_factory=list)
    state_updates: AnalysisFilters | None = None
    clarification: str | None = None


class EvidenceMetric(BaseModel):
    """A display-ready metric sourced directly from deterministic tool output."""

    model_config = ConfigDict(frozen=True)

    label: str
    value: str
    source_tool: str


class EvidenceFinding(BaseModel):
    """Concise deterministic finding and its evidence source."""

    model_config = ConfigDict(frozen=True)

    statement: str
    source_tools: list[str]


EvidenceAtomKind = Literal["metric", "finding", "knowledge", "limitation"]
EvidenceAuthority = Literal["deterministic", "dataset_rule", "engineering_reference", "system_limit"]


class EvidenceAtom(BaseModel):
    """One backend-owned, traceable fact available to an AI answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[MFKL]\d+$")
    kind: EvidenceAtomKind
    statement: str = Field(min_length=1, max_length=2_000)
    value: float | None = None
    display_value: str | None = None
    unit: str | None = None
    source: str = Field(min_length=1)
    authority: EvidenceAuthority = "deterministic"
    scenario_id: str | None = None
    cycle: int | None = Field(default=None, ge=0)
    incident_id: str | None = None


class GroundedStatement(BaseModel):
    """LLM prose paired with the deterministic atoms that support it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=900)
    claim_ids: list[str] = Field(min_length=1, max_length=6)


class GroundedCopilotAnswer(BaseModel):
    """Structured response shape; raw free-form LLM prose is never rendered directly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: GroundedStatement
    evidence: list[GroundedStatement] = Field(default_factory=list, max_length=4)
    next_checks: list[GroundedStatement] = Field(default_factory=list, max_length=3)
    limitations: list[GroundedStatement] = Field(default_factory=list, max_length=3)


class InvestigationTrace(BaseModel):
    """Inspectable orchestration facts, never hidden chain-of-thought."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    scenario_id: str | None = None
    cycle: int | None = Field(default=None, ge=0)
    incident_id: str | None = None
    objective: str
    answerability: Literal["supported", "partially_supported", "unsupported"]
    evidence_needed: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_sources: list[dict[str, str]] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    tool_round_count: int = Field(default=0, ge=0, le=2)
    planner_status: str = "fallback"
    grounding_status: str = "not_run"
    prompt_version: str = "agent-v1"
    knowledge_corpus_version: str = "knowledge-v1"


class EvidencePackage(BaseModel):
    """All numerical/engineering evidence assembled before any later LLM explanation."""

    model_config = ConfigDict(frozen=True)

    question: str
    intent: Intent
    filters: AnalysisFilters
    sample_size: int | None = Field(default=None, ge=0)
    calculations_run: list[str]
    findings: list[EvidenceFinding]
    metrics: list[EvidenceMetric]
    visualisations: list[str]
    statistical_tests: list[dict[str, Any]]
    model_evidence: list[dict[str, Any]]
    engineering_evidence: list[dict[str, Any]]
    data_quality_warnings: list[str]
    uncertainty: list[str]
    limitations: list[str]
    suggested_next_questions: list[str]
    tool_results: dict[str, Any]
    knowledge_evidence: list[EvidenceAtom] = Field(default_factory=list)
    claim_ledger: list[EvidenceAtom] = Field(default_factory=list)


class OfflineCopilotResponse(BaseModel):
    """Deterministic evidence response with an optional non-numerical Gemini interpretation."""

    model_config = ConfigDict(frozen=True)

    answer: str
    evidence: EvidencePackage
    state: dict[str, Any]
    llm_explanation: str | None = None
    llm_status: str = "disabled"
    llm_warning: str | None = None
