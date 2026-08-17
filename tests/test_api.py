"""HTTP-level checks for the FastAPI transport layer using an offline injected service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from industrial_copilot.api.main import create_app
from industrial_copilot.copilot.service import IndustrialCopilotService


def build_client(sample_ai4i_frame: pd.DataFrame, tmp_path: Path) -> TestClient:
    service = IndustrialCopilotService(
        frame=sample_ai4i_frame,
        models_directory=tmp_path,
        llm_enabled=False,
    )
    return TestClient(create_app(service))


def test_health_summary_and_audit_endpoints(sample_ai4i_frame: pd.DataFrame, tmp_path: Path) -> None:
    client = build_client(sample_ai4i_frame, tmp_path)

    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/dataset/summary").json()["observation_count"] == 5
    assert client.get("/audit").json()["row_count"] == 5


def test_analyze_preserves_context_and_available_data_limits(
    sample_ai4i_frame: pd.DataFrame, tmp_path: Path
) -> None:
    client = build_client(sample_ai4i_frame, tmp_path)
    first = client.post("/analyze", json={"question": "What percentage failed?"})
    filtered = client.post(
        "/analyze",
        json={"question": "Only L products.", "state": first.json()["state"]},
    )
    unavailable = client.post(
        "/investigate", json={"question": "What happened during the last 30 days?"}
    )

    assert first.status_code == status_code_ok
    assert filtered.json()["evidence"]["filters"]["product_types"] == ["L"]
    assert unavailable.json()["evidence"]["intent"] == "unavailable_data"


def test_observation_similarity_and_prediction_errors_are_translated(
    sample_ai4i_frame: pd.DataFrame, tmp_path: Path
) -> None:
    client = build_client(sample_ai4i_frame, tmp_path)

    assert client.get("/observation/2").json()["uid"] == 2
    assert client.post("/similar", json={"uid": 2, "k": 3}).json()["target_uid"] == 2
    assert client.get("/observation/999").status_code == 404
    assert (
        client.post(
            "/predict",
            json={
                "observation": {
                    "product_type": "L",
                    "air_temperature_k": 300,
                    "process_temperature_k": 310,
                    "rotational_speed_rpm": 1500,
                    "torque_nm": 40,
                    "tool_wear_min": 10,
                }
            },
        ).status_code
        == 503
    )


def test_live_scenarios_switch_cleanly_and_can_return_to_osf(
    sample_ai4i_frame: pd.DataFrame, tmp_path: Path
) -> None:
    client = build_client(sample_ai4i_frame, tmp_path)
    expected = {
        "OSF": ("INCIDENT", True),
        "HDF": ("WARNING", True),
        "PWF": ("WARNING", True),
    }

    for scenario, (machine_status, has_incident) in expected.items():
        response = client.post("/live/state", json={"scenario": scenario, "cycle": 36})
        body = response.json()
        assert response.status_code == status_code_ok
        assert body["cycle"] == 36
        assert len(body["history"]) == 36
        assert body["status"] == machine_status
        assert (body["incident"] is not None) is has_incident

    replay = client.post("/live/state", json={"scenario": "REPLAY", "cycle": 36}).json()
    assert replay["cycle"] == len(sample_ai4i_frame)
    assert len(replay["history"]) == len(sample_ai4i_frame)

    restarted = client.post("/live/state", json={"scenario": "OSF", "cycle": 1}).json()
    assert restarted["cycle"] == 1
    assert len(restarted["history"]) == 1
    assert restarted["status"] == "NORMAL"
    assert restarted["incident"] is None


def test_live_what_if_uses_selected_scenario_and_consistent_display_status(
    sample_ai4i_frame: pd.DataFrame, tmp_path: Path
) -> None:
    client = build_client(sample_ai4i_frame, tmp_path)

    response = client.post(
        "/live/what-if",
        json={
            "scenario": "OSF",
            "cycle": 36,
            "air_temperature_k": 302.3,
            "process_temperature_k": 308.5,
            "rotational_speed_rpm": 1440,
            "torque_nm": 56,
            "tool_wear_min": 142,
        },
    )
    body = response.json()

    assert response.status_code == status_code_ok
    assert body["scenario"] == "OSF"
    assert body["cycle"] == 36
    assert body["current"]["status"] == "INCIDENT"
    assert body["proposed"]["status"] == "NORMAL"
    assert body["proposed"]["osfMargin"] == 3048
    assert "INCIDENT to NORMAL" in body["summary"]


def test_live_copilot_returns_bounded_trace_and_safe_fallback(
    sample_ai4i_frame: pd.DataFrame, tmp_path: Path
) -> None:
    client = build_client(sample_ai4i_frame, tmp_path)
    response = client.post(
        "/live/copilot",
        json={
            "scenario": "OSF",
            "cycle": 36,
            "question": "The spindle seems stressed as wear builds. What should I inspect?",
            "mode": "deep",
        },
    )
    body = response.json()

    assert response.status_code == status_code_ok
    assert body["ai_generated"] is False
    assert body["investigation_trace"]["planner_status"] == "fallback"
    assert body["investigation_trace"]["tool_round_count"] <= 2
    assert body["citations"]
    assert body["answer_mode"] == "deep"


status_code_ok = 200
