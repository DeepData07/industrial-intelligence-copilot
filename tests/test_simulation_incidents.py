from __future__ import annotations

import numpy as np
import pytest

from industrial_copilot.ml.train import FittedRiskModel
from industrial_copilot.simulation.incidents import IncidentEngine, MonitoringPolicy
from industrial_copilot.simulation.scenarios import generate_osf_scenario
from industrial_copilot.simulation.schemas import SimulationSession
from industrial_copilot.simulation.state import OperationalTwinBuilder


class _FixedEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, _frame):
        return np.array([[1 - self.probability, self.probability]])


def test_incident_engine_does_not_create_false_incident_for_stable_start() -> None:
    events = generate_osf_scenario(cycles=12)[:4]
    state = OperationalTwinBuilder().build(_session_with_history(events))

    result = IncidentEngine().evaluate(state)

    assert result.incident is None
    assert result.created_new_incident is False
    assert result.evidence == ()


def test_incident_engine_creates_warning_from_real_osf_margin() -> None:
    events = generate_osf_scenario(cycles=12)
    state = OperationalTwinBuilder().build(_session_with_history(events))

    result = IncidentEngine().evaluate(state)

    assert result.incident is not None
    assert result.created_new_incident is True
    assert result.incident.incident_id == "INC-0001"
    assert result.incident.severity == "WARNING"
    assert result.incident.first_cycle == len(events)
    assert "Overstrain" in result.incident.primary_reason
    assert "OSF" in result.incident.context.triggered_rules
    assert result.incident.context.current_telemetry.source == "synthetic_demo_scenario"


def test_incident_engine_debounces_duplicate_active_incidents() -> None:
    events = generate_osf_scenario(cycles=12)
    engine = IncidentEngine()

    first = engine.evaluate(OperationalTwinBuilder().build(_session_with_history(events[:11])))
    second = engine.evaluate(OperationalTwinBuilder().build(_session_with_history(events)))

    assert first.incident is not None
    assert second.incident is not None
    assert first.created_new_incident is True
    assert second.created_new_incident is False
    assert second.incident.incident_id == first.incident.incident_id
    assert second.incident.latest_cycle == len(events)


def test_incident_engine_clears_and_resets() -> None:
    events = generate_osf_scenario(cycles=12)
    engine = IncidentEngine()
    warning = engine.evaluate(OperationalTwinBuilder().build(_session_with_history(events)))
    assert warning.incident is not None

    normal = engine.evaluate(OperationalTwinBuilder().build(_session_with_history(events[:3])))
    assert normal.incident is None
    assert normal.cleared_active_incident is True

    engine.evaluate(OperationalTwinBuilder().build(_session_with_history(events)))
    assert engine.active_incident is not None
    engine.reset()
    assert engine.active_incident is None
    repeated = engine.evaluate(OperationalTwinBuilder().build(_session_with_history(events)))
    assert repeated.incident is not None
    assert repeated.incident.incident_id == "INC-0001"


def test_incident_engine_escalates_from_model_risk_policy() -> None:
    events = generate_osf_scenario(cycles=3)
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
        estimator=_FixedEstimator(0.41),  # type: ignore[arg-type]
    )
    twin = OperationalTwinBuilder(risk_model=model).build(_session_with_history(events))
    engine = IncidentEngine(
        MonitoringPolicy(warning_risk_threshold=0.20, incident_risk_threshold=0.35)
    )

    result = engine.evaluate(twin)

    assert result.incident is not None
    assert result.incident.severity == "INCIDENT"
    assert result.incident.context.failure_risk_probability == pytest.approx(0.41)
    assert "Failure risk" in result.incident.primary_reason


def _session_with_history(events) -> SimulationSession:
    return SimulationSession(
        session_id=events[0].simulation_session_id,
        asset_id=events[0].asset_id,
        source="synthetic_demo_scenario",
        source_label=events[0].source_label,
        status="running",
        cursor_index=len(events),
        total_cycles=12,
        history=tuple(events),
    )
