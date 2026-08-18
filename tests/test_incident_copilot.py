from __future__ import annotations

from industrial_copilot.copilot.service import IndustrialCopilotService
from industrial_copilot.copilot.state import ConversationState
from industrial_copilot.data.schema import ROTATIONAL_SPEED
from industrial_copilot.simulation.incidents import IncidentEngine
from industrial_copilot.simulation.investigation import build_incident_investigation_package
from industrial_copilot.simulation.scenarios import generate_osf_scenario
from industrial_copilot.simulation.schemas import SimulationSession
from industrial_copilot.simulation.state import OperationalTwinBuilder


def test_incident_copilot_explains_why_alert_was_flagged(sample_ai4i_frame) -> None:
    package = _osf_incident_package(sample_ai4i_frame)
    service = IndustrialCopilotService(frame=sample_ai4i_frame, llm_enabled=False)

    response = service.ask_about_incident("Why did you flag this?", package)

    assert response.evidence.intent == "incident_investigation"
    assert response.state["current_incident_id"] == package.incident_id
    assert "Largest recent change" in response.answer
    assert "Similar historical conditions" in response.answer
    assert "calculate_what_changed" in response.evidence.calculations_run
    assert "find_similar_historical_conditions_for_event" in response.evidence.calculations_run


def test_incident_copilot_answers_similar_history_follow_up(sample_ai4i_frame) -> None:
    package = _osf_incident_package(sample_ai4i_frame)
    service = IndustrialCopilotService(frame=sample_ai4i_frame, llm_enabled=False)

    response = service.ask_about_incident("Has this happened under similar conditions before?", package)

    assert "Similar historical conditions" in response.answer
    assert response.evidence.metrics[-1].label == "Similar-case failure rate"
    assert response.evidence.tool_results["incident_investigation_package"]["incident_id"] == package.incident_id


def test_incident_copilot_tests_rpm_hypothesis_against_current_window(sample_ai4i_frame) -> None:
    package = _osf_incident_package(sample_ai4i_frame)
    service = IndustrialCopilotService(frame=sample_ai4i_frame, llm_enabled=False)

    response = service.ask_about_incident("Is RPM causing this?", package)

    assert ROTATIONAL_SPEED in response.answer
    assert "does not support RPM as the primary driver" in response.answer
    assert response.evidence.findings[0].source_tools == ["calculate_what_changed"]


def test_incident_copilot_calculates_rule_based_adjustment_options(sample_ai4i_frame) -> None:
    package = _osf_incident_package(sample_ai4i_frame)
    service = IndustrialCopilotService(frame=sample_ai4i_frame, llm_enabled=False)

    response = service.ask_about_incident(
        "What parameter should I change to resolve this and by how much?",
        package,
    )

    assert package.adjustment_options
    torque = package.adjustment_options[0]
    assert torque.parameter == "Torque"
    assert torque.proposed_value < torque.current_value
    assert torque.expected_osf_margin_min_nm >= 1000
    assert "reduce torque from" in response.answer
    assert "What-if analysis" in response.answer
    assert response.evidence.findings[0].source_tools[0] == "calculate_rule_based_adjustment"


def test_incident_copilot_keeps_normal_dataset_flow_unchanged(sample_ai4i_frame) -> None:
    package = _osf_incident_package(sample_ai4i_frame)
    service = IndustrialCopilotService(frame=sample_ai4i_frame, llm_enabled=False)

    incident_response = service.ask_about_incident("Why did you flag this?", package)
    normal_response = service.ask(
        "What percentage failed?",
        ConversationState.model_validate(incident_response.state),
    )

    assert normal_response.evidence.intent == "failure_rate"
    assert "selected observations failed" in normal_response.answer
    assert normal_response.state["current_incident_id"] == package.incident_id


def _osf_incident_package(sample_ai4i_frame):
    events = generate_osf_scenario(cycles=12)
    twin = OperationalTwinBuilder().build(
        SimulationSession(
            session_id=events[0].simulation_session_id,
            asset_id=events[0].asset_id,
            source="synthetic_demo_scenario",
            source_label=events[0].source_label,
            status="running",
            cursor_index=len(events),
            total_cycles=len(events),
            history=events,
        )
    )
    incident = IncidentEngine().evaluate(twin).incident
    assert incident is not None
    return build_incident_investigation_package(
        incident,
        twin,
        sample_ai4i_frame,
        recent_window_size=4,
        baseline_window_size=4,
        similar_k=3,
    )
