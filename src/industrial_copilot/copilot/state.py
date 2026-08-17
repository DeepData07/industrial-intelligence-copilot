"""Compact, explicit conversation state retained across contextual follow-up questions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.analytics.models import AnalysisFilters
from industrial_copilot.copilot.schemas import Intent


class ConversationState(BaseModel):
    """Only the context needed for a safe deterministic follow-up; never private reasoning."""

    model_config = ConfigDict(frozen=True)

    current_filters: AnalysisFilters = Field(default_factory=AnalysisFilters)
    current_product_type: str | None = None
    current_uid: int | None = None
    current_comparison: str | None = None
    current_variable: str | None = None
    current_failure_mode: str | None = None
    current_incident_id: str | None = None
    previous_intent: Intent | None = None
    previous_result: dict[str, Any] | None = None


def updated_state(
    previous: ConversationState,
    intent: Intent,
    filters: AnalysisFilters,
    result: dict[str, Any] | None = None,
    variable: str | None = None,
    uid: int | None = None,
    incident_id: str | None = None,
) -> ConversationState:
    """Return an immutable-style new state after a completed deterministic investigation."""

    return ConversationState(
        current_filters=filters,
        current_product_type=(filters.product_types or [None])[0],
        current_uid=uid if uid is not None else previous.current_uid,
        current_comparison="failed_vs_healthy" if intent == "failed_healthy_comparison" else None,
        current_variable=variable or previous.current_variable,
        current_failure_mode=filters.failure_mode,
        current_incident_id=incident_id or previous.current_incident_id,
        previous_intent=intent,
        previous_result=result,
    )
