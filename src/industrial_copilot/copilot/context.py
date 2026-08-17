"""Deterministic context resolution for bounded Copilot follow-ups."""

from __future__ import annotations

from industrial_copilot.copilot.agent_schemas import StructuredInvestigationContext
from industrial_copilot.copilot.state import ConversationState


def resolve_live_context(
    *, scenario: str, cycle: int, incident_id: str | None, state: ConversationState | None = None
) -> StructuredInvestigationContext:
    """Turn retained public state into a small planner context; no hidden reasoning is stored."""

    prior = state or ConversationState()
    variables = [item for item in (prior.current_variable,) if item]
    unavailable = [
        "timestamped plant history", "vibration", "audio", "maintenance history",
        "live PLC/SCADA telemetry", "remaining useful life", "machine-command capability",
    ]
    return StructuredInvestigationContext(
        scenario_id=scenario,
        cycle=cycle,
        incident_id=incident_id,
        previous_intent=prior.previous_intent,
        active_filters=prior.current_filters.model_dump(),
        active_variables=variables,
        last_comparison=prior.current_comparison,
        available_data=[
            "current simulated telemetry", "calibrated model risk", "engineering rule margins",
            "recent baseline comparison", "AI4I similar conditions",
        ],
        unavailable_data=unavailable,
    )
