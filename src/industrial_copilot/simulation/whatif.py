"""Deterministic what-if analysis for simulated operating conditions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from industrial_copilot.ml.train import FittedRiskModel
from industrial_copilot.simulation.schemas import SimulationSession, TelemetryEvent
from industrial_copilot.simulation.state import OperationalTwinBuilder, OperationalTwinState


class WhatIfInput(BaseModel):
    """Optional slider-style overrides for one proposed operating point."""

    model_config = ConfigDict(frozen=True)

    air_temperature_k: float | None = Field(default=None, gt=0)
    process_temperature_k: float | None = Field(default=None, gt=0)
    rotational_speed_rpm: float | None = Field(default=None, gt=0)
    torque_nm: float | None = Field(default=None, ge=0)
    tool_wear_min: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_at_least_one_change(self) -> WhatIfInput:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("At least one proposed operating value is required.")
        return self


class WhatIfResult(BaseModel):
    """Current/proposed comparison for UI sliders and incident decision support."""

    model_config = ConfigDict(frozen=True)

    current_state: OperationalTwinState
    proposed_state: OperationalTwinState
    changed_fields: tuple[str, ...]
    summary: str
    limitations: tuple[str, ...]


class WhatIfEngine:
    """Recalculate proposed telemetry through the same twin/risk/rule pipeline."""

    def __init__(self, risk_model: FittedRiskModel | None = None) -> None:
        self._builder = OperationalTwinBuilder(risk_model=risk_model)

    def evaluate(self, session: SimulationSession, proposed: WhatIfInput) -> WhatIfResult:
        """Compare current session state with a proposed slider-adjusted telemetry point."""

        if not session.history:
            raise ValueError("Cannot run what-if analysis before telemetry has been emitted.")
        current_event = session.history[-1]
        proposed_event = _proposed_event(current_event, proposed)
        current_state = self._builder.build(session)
        proposed_session = session.model_copy(
            update={
                "history": (*session.history[:-1], proposed_event),
                "status": "running" if session.status != "complete" else session.status,
            }
        )
        proposed_state = self._builder.build(proposed_session)
        changed_fields = _changed_fields(current_event, proposed_event)
        return WhatIfResult(
            current_state=current_state,
            proposed_state=proposed_state,
            changed_fields=changed_fields,
            summary=_summary(current_state, proposed_state, changed_fields),
            limitations=(
                "What-if analysis is decision support only; no machine command has been issued.",
                "Risk and rule changes are calculated from simulated/proposed telemetry, not guaranteed outcomes.",
            ),
        )


def _proposed_event(current: TelemetryEvent, proposed: WhatIfInput) -> TelemetryEvent:
    return current.model_copy(
        update={
            "uid": None,
            "product_id": None,
            "air_temperature_k": proposed.air_temperature_k or current.air_temperature_k,
            "process_temperature_k": proposed.process_temperature_k or current.process_temperature_k,
            "rotational_speed_rpm": proposed.rotational_speed_rpm or current.rotational_speed_rpm,
            "torque_nm": proposed.torque_nm if proposed.torque_nm is not None else current.torque_nm,
            "tool_wear_min": proposed.tool_wear_min if proposed.tool_wear_min is not None else current.tool_wear_min,
        }
    )


def _changed_fields(current: TelemetryEvent, proposed: TelemetryEvent) -> tuple[str, ...]:
    fields = []
    for field in (
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ):
        if getattr(current, field) != getattr(proposed, field):
            fields.append(field)
    return tuple(fields)


def _summary(
    current_state: OperationalTwinState,
    proposed_state: OperationalTwinState,
    changed_fields: tuple[str, ...],
) -> str:
    if not changed_fields:
        return "The proposed state matches the current telemetry."
    current_margin = (
        current_state.rule_margins.osf_remaining_margin_min_nm
        if current_state.rule_margins is not None
        else None
    )
    proposed_margin = (
        proposed_state.rule_margins.osf_remaining_margin_min_nm
        if proposed_state.rule_margins is not None
        else None
    )
    if current_margin is not None and proposed_margin is not None:
        return (
            f"Proposed changes update status from {current_state.machine_status} to "
            f"{proposed_state.machine_status}; OSF margin changes from {current_margin:.0f} "
            f"to {proposed_margin:.0f} min Nm."
        )
    return (
        f"Proposed changes update status from {current_state.machine_status} to "
        f"{proposed_state.machine_status}."
    )
