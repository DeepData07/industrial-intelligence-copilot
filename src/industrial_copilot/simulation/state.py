"""Operational twin state derived from simulated telemetry and existing evidence logic."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
)
from industrial_copilot.features.engineering import (
    HDF_RULE_CONDITION,
    MECHANICAL_POWER,
    OSF_RULE_CONDITION,
    OVERSTRAIN_LOAD,
    OVERSTRAIN_THRESHOLD,
    PWF_RULE_CONDITION,
    TEMPERATURE_DELTA,
    calculate_operating_features,
)
from industrial_copilot.ml.predict import get_model_risk
from industrial_copilot.ml.schemas import PredictionInput, RiskPrediction
from industrial_copilot.ml.train import FittedRiskModel
from industrial_copilot.simulation.schemas import SimulationSession, TelemetryEvent

MachineStatus = Literal["NORMAL", "WATCH", "WARNING", "INCIDENT"]


class RuleMargins(BaseModel):
    """Interpretable proximity to documented AI4I operating envelopes."""

    model_config = ConfigDict(frozen=True)

    osf_remaining_margin_min_nm: float
    hdf_temperature_delta_margin_k: float
    hdf_rpm_margin: float
    pwf_low_power_margin_w: float
    pwf_high_power_margin_w: float
    triggered_rules: tuple[str, ...]


class EngineeredTelemetry(BaseModel):
    """Current telemetry plus deterministic engineering fields."""

    model_config = ConfigDict(frozen=True)

    temperature_delta_k: float
    mechanical_power_w: float
    overstrain_load_min_nm: float
    overstrain_threshold_min_nm: float
    hdf_rule_condition: bool
    pwf_rule_condition: bool
    osf_rule_condition: bool


class OperationalTwinState(BaseModel):
    """Current machine-twin state for live UI, incidents, and later copilot context."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    asset_id: str
    source: str
    source_label: str
    simulation_status: str
    machine_status: MachineStatus
    current_cycle: int
    total_cycles: int
    current_telemetry: TelemetryEvent | None
    engineered: EngineeredTelemetry | None
    rule_margins: RuleMargins | None
    risk: RiskPrediction | None
    risk_note: str
    recent_history: tuple[TelemetryEvent, ...] = ()
    limitations: tuple[str, ...] = Field(default_factory=tuple)


class OperationalTwinBuilder:
    """Build live machine state without mutating simulation sessions or telemetry events."""

    def __init__(self, risk_model: FittedRiskModel | None = None) -> None:
        self._risk_model = risk_model

    def build(self, session: SimulationSession) -> OperationalTwinState:
        """Return an operational twin snapshot from the latest session history."""

        current = session.history[-1] if session.history else None
        engineered = _engineered_for_event(current) if current is not None else None
        margins = _rule_margins(current, engineered) if current is not None and engineered is not None else None
        risk, risk_note = self._risk_for_event(current)
        return OperationalTwinState(
            session_id=session.session_id,
            asset_id=session.asset_id,
            source=session.source,
            source_label=session.source_label,
            simulation_status=session.status,
            machine_status=_machine_status(session.status, engineered, risk),
            current_cycle=session.current_cycle,
            total_cycles=session.total_cycles,
            current_telemetry=current,
            engineered=engineered,
            rule_margins=margins,
            risk=risk,
            risk_note=risk_note,
            recent_history=session.history,
            limitations=(
                "Telemetry is simulated from AI4I replay or a disclosed synthetic scenario.",
                "Failure risk is decision-support evidence, not remaining useful life.",
            ),
        )

    def _risk_for_event(self, event: TelemetryEvent | None) -> tuple[RiskPrediction | None, str]:
        if event is None:
            return None, "No telemetry has been emitted for this session yet."
        if self._risk_model is None:
            return None, "No local risk model was supplied to the operational twin builder."
        return (
            get_model_risk(
                self._risk_model,
                PredictionInput(
                    product_type=event.product_type,
                    air_temperature_k=event.air_temperature_k,
                    process_temperature_k=event.process_temperature_k,
                    rotational_speed_rpm=event.rotational_speed_rpm,
                    torque_nm=event.torque_nm,
                    tool_wear_min=event.tool_wear_min,
                ),
            ),
            "Calibrated local model risk was calculated from current telemetry.",
        )


def telemetry_events_to_operating_frame(events: tuple[TelemetryEvent, ...]) -> pd.DataFrame:
    """Convert live telemetry history into the operating frame used by analytics."""

    return pd.DataFrame(
        [
            {
                PRODUCT_TYPE: event.product_type,
                AIR_TEMPERATURE: event.air_temperature_k,
                PROCESS_TEMPERATURE: event.process_temperature_k,
                ROTATIONAL_SPEED: event.rotational_speed_rpm,
                TORQUE: event.torque_nm,
                TOOL_WEAR: event.tool_wear_min,
            }
            for event in events
        ]
    )


def _engineered_for_event(event: TelemetryEvent) -> EngineeredTelemetry:
    engineered = calculate_operating_features(telemetry_events_to_operating_frame((event,))).iloc[0]
    return EngineeredTelemetry(
        temperature_delta_k=float(engineered[TEMPERATURE_DELTA]),
        mechanical_power_w=float(engineered[MECHANICAL_POWER]),
        overstrain_load_min_nm=float(engineered[OVERSTRAIN_LOAD]),
        overstrain_threshold_min_nm=float(engineered[OVERSTRAIN_THRESHOLD]),
        hdf_rule_condition=bool(engineered[HDF_RULE_CONDITION]),
        pwf_rule_condition=bool(engineered[PWF_RULE_CONDITION]),
        osf_rule_condition=bool(engineered[OSF_RULE_CONDITION]),
    )


def _rule_margins(event: TelemetryEvent, engineered: EngineeredTelemetry) -> RuleMargins:
    triggered = []
    if engineered.hdf_rule_condition:
        triggered.append("HDF")
    if engineered.pwf_rule_condition:
        triggered.append("PWF")
    if engineered.osf_rule_condition:
        triggered.append("OSF")
    return RuleMargins(
        osf_remaining_margin_min_nm=(
            engineered.overstrain_threshold_min_nm - engineered.overstrain_load_min_nm
        ),
        hdf_temperature_delta_margin_k=engineered.temperature_delta_k - 8.6,
        hdf_rpm_margin=event.rotational_speed_rpm - 1380,
        pwf_low_power_margin_w=engineered.mechanical_power_w - 3500,
        pwf_high_power_margin_w=9000 - engineered.mechanical_power_w,
        triggered_rules=tuple(triggered),
    )


def _machine_status(
    _simulation_status: str,
    engineered: EngineeredTelemetry | None,
    risk: RiskPrediction | None,
) -> MachineStatus:
    if engineered is None:
        return "NORMAL"
    # Use the most severe signal. A triggered engineering rule is a warning,
    # but it must not mask an incident-level calibrated risk estimate.
    if risk is not None and risk.failure_probability >= 0.35:
        return "INCIDENT"
    if engineered.hdf_rule_condition or engineered.pwf_rule_condition or engineered.osf_rule_condition:
        return "WARNING"
    if risk is not None:
        if risk.failure_probability >= 0.15:
            return "WARNING"
        if risk.failure_probability >= 0.05:
            return "WATCH"
    return "NORMAL"
