from __future__ import annotations

import numpy as np
import pytest

from industrial_copilot.ml.train import FittedRiskModel
from industrial_copilot.simulation.scenarios import generate_osf_scenario
from industrial_copilot.simulation.schemas import SimulationSession
from industrial_copilot.simulation.whatif import WhatIfEngine, WhatIfInput


class _TorqueSensitiveEstimator:
    def predict_proba(self, frame):
        torque = float(frame["Torque [Nm]"].iloc[0])
        probability = 0.30 if torque >= 50 else 0.08
        return np.array([[1 - probability, probability]])


class _IncidentRiskEstimator:
    def predict_proba(self, frame):
        probability = 0.75
        return np.array([[1 - probability, probability]])


def test_whatif_lowering_torque_recalculates_margin_and_status() -> None:
    events = generate_osf_scenario(cycles=12)
    session = _session_with_history(events)

    result = WhatIfEngine().evaluate(session, WhatIfInput(torque_nm=42.0))

    assert result.current_state.rule_margins is not None
    assert result.proposed_state.rule_margins is not None
    assert result.current_state.rule_margins.osf_remaining_margin_min_nm < 0
    assert result.proposed_state.rule_margins.osf_remaining_margin_min_nm > 0
    assert result.current_state.machine_status == "WARNING"
    assert result.proposed_state.machine_status == "NORMAL"
    assert result.changed_fields == ("torque_nm",)
    assert "no machine command" in result.limitations[0]


def test_whatif_keeps_source_honest_and_does_not_mutate_current_event() -> None:
    events = generate_osf_scenario(cycles=12)
    session = _session_with_history(events)
    current = session.history[-1]

    result = WhatIfEngine().evaluate(
        session,
        WhatIfInput(rotational_speed_rpm=1500.0, torque_nm=40.0),
    )

    assert session.history[-1] == current
    assert result.proposed_state.current_telemetry is not None
    assert result.proposed_state.current_telemetry.source == current.source
    assert result.proposed_state.current_telemetry.uid is None
    assert set(result.changed_fields) == {"rotational_speed_rpm", "torque_nm"}


def test_whatif_recalculates_model_risk_when_model_is_available() -> None:
    events = generate_osf_scenario(cycles=12)
    model = FittedRiskModel(
        model_name="random_forest",
        feature_set="engineering_augmented",
        input_features=(
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Temperature delta [K]",
            "Mechanical power [W]",
            "Overstrain load [min Nm]",
        ),
        estimator=_TorqueSensitiveEstimator(),  # type: ignore[arg-type]
    )

    result = WhatIfEngine(risk_model=model).evaluate(
        _session_with_history(events),
        WhatIfInput(torque_nm=42.0),
    )

    assert result.current_state.risk is not None
    assert result.proposed_state.risk is not None
    assert result.current_state.risk.failure_probability == pytest.approx(0.30)
    assert result.proposed_state.risk.failure_probability == pytest.approx(0.08)


def test_whatif_rule_warning_does_not_mask_incident_level_model_risk() -> None:
    events = generate_osf_scenario(cycles=12)
    model = FittedRiskModel(
        model_name="random_forest",
        feature_set="engineering_augmented",
        input_features=(
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Temperature delta [K]",
            "Mechanical power [W]",
            "Overstrain load [min Nm]",
        ),
        estimator=_IncidentRiskEstimator(),  # type: ignore[arg-type]
    )

    result = WhatIfEngine(risk_model=model).evaluate(
        _session_with_history(events),
        WhatIfInput(torque_nm=events[-1].torque_nm),
    )

    assert result.current_state.machine_status == "INCIDENT"
    assert result.proposed_state.machine_status == "INCIDENT"


def test_whatif_rejects_missing_telemetry_and_empty_proposal() -> None:
    with pytest.raises(ValueError, match="At least one proposed"):
        WhatIfInput()

    empty = SimulationSession(
        session_id="SIM-EMPTY",
        asset_id="MACHINE-01",
        source="synthetic_demo_scenario",
        source_label="Synthetic Demo Scenario Based on AI4I Rules",
        total_cycles=0,
    )
    with pytest.raises(ValueError, match="before telemetry"):
        WhatIfEngine().evaluate(empty, WhatIfInput(torque_nm=42.0))


def _session_with_history(events) -> SimulationSession:
    return SimulationSession(
        session_id=events[0].simulation_session_id,
        asset_id=events[0].asset_id,
        source="synthetic_demo_scenario",
        source_label=events[0].source_label,
        status="running",
        cursor_index=len(events),
        total_cycles=len(events),
        history=events,
    )
