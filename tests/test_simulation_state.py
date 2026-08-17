from __future__ import annotations

import numpy as np
import pytest

from industrial_copilot.ml.train import FittedRiskModel
from industrial_copilot.simulation.replay import AI4IReplayEngine
from industrial_copilot.simulation.scenarios import generate_hdf_scenario, generate_osf_scenario
from industrial_copilot.simulation.schemas import SimulationSession
from industrial_copilot.simulation.state import OperationalTwinBuilder


class _FixedEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, _frame):
        return np.array([[1 - self.probability, self.probability]])


def test_operational_twin_handles_empty_session() -> None:
    session = SimulationSession(
        session_id="SIM-EMPTY",
        asset_id="MACHINE-01",
        source="ai4i_replay",
        source_label="AI4I Dataset Replay",
        total_cycles=10,
    )

    state = OperationalTwinBuilder().build(session)

    assert state.machine_status == "NORMAL"
    assert state.current_telemetry is None
    assert state.engineered is None
    assert state.rule_margins is None
    assert state.risk is None
    assert "No telemetry" in state.risk_note


def test_operational_twin_builds_replay_state_with_engineering_and_history(sample_ai4i_frame) -> None:
    engine = AI4IReplayEngine(sample_ai4i_frame, rolling_window_size=2)
    engine.start(speed=5)
    engine.next_events_for_tick()

    state = OperationalTwinBuilder().build(engine.session)

    assert state.session_id == engine.session.session_id
    assert state.source == "ai4i_replay"
    assert state.current_cycle == len(sample_ai4i_frame)
    assert state.current_telemetry is not None
    assert state.engineered is not None
    assert state.rule_margins is not None
    assert len(state.recent_history) == 2
    assert state.simulation_status == "complete"


def test_completed_stream_status_still_comes_from_machine_evidence() -> None:
    event = generate_osf_scenario(cycles=12)[0]
    session = SimulationSession(
        session_id=event.simulation_session_id,
        asset_id=event.asset_id,
        source=event.source,
        source_label=event.source_label,
        status="complete",
        cursor_index=1,
        total_cycles=1,
        history=(event,),
    )

    state = OperationalTwinBuilder().build(session)

    assert state.simulation_status == "complete"
    assert state.machine_status == "NORMAL"


def test_operational_twin_surfaces_rule_margins_from_real_scenario_features() -> None:
    events = generate_osf_scenario(cycles=12)
    session = _session_with_history(events)

    state = OperationalTwinBuilder().build(session)

    assert state.engineered is not None
    assert state.rule_margins is not None
    assert state.machine_status == "WARNING"
    assert state.engineered.osf_rule_condition is True
    assert "OSF" in state.rule_margins.triggered_rules
    assert state.rule_margins.osf_remaining_margin_min_nm < 0


def test_operational_twin_reports_hdf_margins() -> None:
    events = generate_hdf_scenario(cycles=12)
    state = OperationalTwinBuilder().build(_session_with_history(events))

    assert state.engineered is not None
    assert state.rule_margins is not None
    assert state.engineered.hdf_rule_condition is True
    assert state.rule_margins.hdf_temperature_delta_margin_k < 0
    assert state.rule_margins.hdf_rpm_margin < 0
    assert "HDF" in state.rule_margins.triggered_rules


def test_operational_twin_can_include_model_risk() -> None:
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
        estimator=_FixedEstimator(0.22),  # type: ignore[arg-type]
    )

    state = OperationalTwinBuilder(risk_model=model).build(_session_with_history(events))

    assert state.risk is not None
    assert state.risk.failure_probability == pytest.approx(0.22)
    assert state.risk.risk_level == "HIGH_RISK"
    assert state.machine_status == "WARNING"
    assert "model risk" in state.risk_note


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
