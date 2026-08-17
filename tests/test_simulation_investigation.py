from __future__ import annotations

import pytest

from industrial_copilot.data.schema import TORQUE
from industrial_copilot.simulation.incidents import IncidentEngine
from industrial_copilot.simulation.investigation import (
    build_incident_investigation_package,
    calculate_what_changed,
    find_similar_historical_conditions_for_event,
)
from industrial_copilot.simulation.scenarios import generate_osf_scenario
from industrial_copilot.simulation.schemas import SimulationSession
from industrial_copilot.simulation.state import OperationalTwinBuilder


def test_what_changed_compares_recent_window_against_baseline() -> None:
    events = generate_osf_scenario(cycles=12)

    result = calculate_what_changed(events, recent_window_size=4, baseline_window_size=4)

    changes_by_feature = {item.feature: item for item in result.changes}
    assert result.baseline_observation_count == 4
    assert result.recent_observation_count == 4
    assert changes_by_feature[TORQUE].direction == "increased"
    assert changes_by_feature[TORQUE].percent_change is not None
    assert changes_by_feature[TORQUE].percent_change > 0
    assert result.largest_changes
    assert "Largest recent change" in result.summary


def test_what_changed_reports_insufficient_history_without_inventing_baseline() -> None:
    events = generate_osf_scenario(cycles=5)

    result = calculate_what_changed(events, recent_window_size=4, baseline_window_size=4)

    assert result.changes == ()
    assert result.largest_changes == ()
    assert "Need at least 8" in result.summary
    assert result.limitations


def test_live_event_similarity_uses_historical_ai4i_evidence(sample_ai4i_frame) -> None:
    event = generate_osf_scenario(cycles=12)[-1]

    result = find_similar_historical_conditions_for_event(sample_ai4i_frame, event, k=3)

    assert result.candidate_count == len(sample_ai4i_frame)
    assert result.returned_observation_count == 3
    assert 0 <= result.failed_observation_count <= 3
    assert result.similar_case_failure_rate is not None
    assert len(result.observations) == 3
    assert all(observation.uid is not None for observation in result.observations)
    assert "associative evidence" in result.note


def test_incident_investigation_package_combines_change_and_similarity(sample_ai4i_frame) -> None:
    events = generate_osf_scenario(cycles=12)
    twin = OperationalTwinBuilder().build(_session_with_history(events))
    incident_result = IncidentEngine().evaluate(twin)
    assert incident_result.incident is not None

    package = build_incident_investigation_package(
        incident_result.incident,
        twin,
        sample_ai4i_frame,
        recent_window_size=4,
        baseline_window_size=4,
        similar_k=3,
    )

    assert package.incident_id == incident_result.incident.incident_id
    assert package.what_changed.largest_changes
    assert package.similar_historical_conditions.returned_observation_count == 3
    assert any("not real timestamped plant events" in item for item in package.limitations)


def test_investigation_rejects_invalid_inputs(sample_ai4i_frame) -> None:
    event = generate_osf_scenario(cycles=3)[-1]

    with pytest.raises(ValueError, match="Window sizes"):
        calculate_what_changed((event,), recent_window_size=0)
    with pytest.raises(ValueError, match="top_n"):
        calculate_what_changed((event,), top_n=0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        find_similar_historical_conditions_for_event(sample_ai4i_frame, event, k=0)


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
