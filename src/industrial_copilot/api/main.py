"""FastAPI endpoints exposing the existing evidence-first copilot without duplicate logic."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Annotated, Any, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from industrial_copilot.api.schemas import (
    AlertRequest,
    AlertResponse,
    HealthResponse,
    LiveCopilotRequest,
    LiveStateRequest,
    LiveWhatIfRequest,
    PredictionRequest,
    QuestionRequest,
    SimilarRequest,
)
from industrial_copilot.config import PROJECT_ROOT, get_settings
from industrial_copilot.copilot.schemas import OfflineCopilotResponse, ToolCall
from industrial_copilot.copilot.service import IndustrialCopilotService
from industrial_copilot.features.engineering import calculate_operating_features
from industrial_copilot.ml.predict import load_fitted_risk_model
from industrial_copilot.ml.schemas import PredictionInput
from industrial_copilot.ml.train import model_artifact_name
from industrial_copilot.simulation import (
    AI4IReplayEngine,
    IncidentEngine,
    OperationalTwinBuilder,
    SimulationSession,
    TelemetryEvent,
    WhatIfEngine,
    WhatIfInput,
    build_incident_investigation_package,
    generate_hdf_scenario,
    generate_osf_scenario,
    generate_pwf_scenario,
    telemetry_events_to_operating_frame,
)
from industrial_copilot.tools.registry import ToolArgumentError, UnknownToolError

Result = TypeVar("Result")


def create_app(service: IndustrialCopilotService | None = None) -> FastAPI:
    """Build an app; a service can be injected for fast, offline API tests."""

    app = FastAPI(
        title="Industrial Intelligence Copilot API",
        version="0.1.0",
        description="Evidence-first AI4I failure investigation. Calculations use registered tools.",
    )
    app.state.copilot_service = service
    app.add_middleware(
        CORSMiddleware,
        # Local development permits the separate Vite origin and accepts no credentials.
        # Production deployments should replace this with an explicit origin allowlist.
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        settings = get_settings()
        return HealthResponse(
            status="ok",
            project_name=settings.project_name,
            llm_enabled=settings.llm_enabled,
            llm_provider=settings.llm_provider,
        )

    @app.get("/dataset/summary", tags=["dataset"])
    def dataset_summary(
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> dict[str, Any]:
        return _run_tool(copilot, ToolCall(name="get_dataset_summary"))

    @app.get("/audit", tags=["dataset"])
    def audit(
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> dict[str, Any]:
        return _run_tool(copilot, ToolCall(name="run_data_contract_audit"))

    @app.post("/analyze", response_model=OfflineCopilotResponse, tags=["investigation"])
    def analyze(
        payload: QuestionRequest,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> OfflineCopilotResponse:
        return copilot.ask(payload.question, payload.state)

    @app.post("/investigate", response_model=OfflineCopilotResponse, tags=["investigation"])
    def investigate(
        payload: QuestionRequest,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> OfflineCopilotResponse:
        """Expose the evidence-first analysis service under an investigation route."""

        return copilot.ask(payload.question, payload.state)

    @app.get("/observation/{uid}", tags=["observations"])
    def observation(
        uid: int,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> dict[str, Any]:
        return _run_tool(copilot, ToolCall(name="get_observation", arguments={"uid": uid}))

    @app.post("/similar", tags=["observations"])
    def similar(
        payload: SimilarRequest,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> dict[str, Any]:
        return _run_tool(
            copilot,
            ToolCall(
                name="find_similar_conditions",
                arguments={
                    "uid": payload.uid,
                    "k": payload.k,
                    "filters": payload.filters.model_dump(),
                },
            ),
        )

    @app.post("/predict", tags=["risk"])
    def predict(
        payload: PredictionRequest,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> dict[str, Any]:
        return _model_risk(copilot, payload.model_name, payload.feature_set, payload.observation)

    @app.post("/explain-alert", response_model=AlertResponse, tags=["risk"])
    def explain_alert(
        payload: AlertRequest,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> AlertResponse:
        observation_record = _run_tool(
            copilot, ToolCall(name="get_observation", arguments={"uid": payload.uid})
        )
        engineering = _run_tool(
            copilot, ToolCall(name="calculate_engineering_features", arguments={"uid": payload.uid})
        )
        similar_cases = _run_tool(
            copilot,
            ToolCall(name="find_similar_conditions", arguments={"uid": payload.uid, "k": payload.k}),
        )
        observation = PredictionInput(
            product_type=str(observation_record["values"]["Type"]),
            air_temperature_k=float(observation_record["values"]["Air temperature [K]"]),
            process_temperature_k=float(observation_record["values"]["Process temperature [K]"]),
            rotational_speed_rpm=float(observation_record["values"]["Rotational speed [rpm]"]),
            torque_nm=float(observation_record["values"]["Torque [Nm]"]),
            tool_wear_min=float(observation_record["values"]["Tool wear [min]"]),
        )
        risk = _model_risk(copilot, payload.model_name, payload.feature_set, observation)
        return AlertResponse(
            observation=observation_record,
            engineering_features=engineering,
            similar_conditions=similar_cases,
            model_risk=risk,
        )

    @app.post("/live/state", tags=["live operations"])
    def live_state(
        payload: LiveStateRequest,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> dict[str, Any]:
        """Return a real operational snapshot for the React console."""

        snapshot = _live_snapshot(copilot, payload.scenario, payload.cycle)
        return _serialize_live_snapshot(snapshot)

    @app.post("/live/copilot", tags=["live operations"])
    def live_copilot(
        payload: LiveCopilotRequest,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> dict[str, Any]:
        """Answer a live-incident question using the existing evidence-first service."""

        snapshot = _live_snapshot(copilot, payload.scenario, payload.cycle)
        package = snapshot["package"]
        if package is None:
            twin = snapshot["twin"]
            current = _serialize_current(twin)
            status_label = current["status"]
            risk_percent = current["risk"] * 100
            answer = (
                f"{payload.scenario} is the active scenario at cycle {current['cycle']}. "
                f"The current machine status is {status_label}, with a calibrated failure-risk "
                f"estimate of {risk_percent:.1f}%. No active incident is open for this scenario."
            )
            return {
                "scenario": payload.scenario,
                "cycle": payload.cycle,
                "answer": answer,
                "verified_answer": answer,
                "ai_generated": False,
                "ai_status": "not_applicable",
                "ai_provider": None,
                "ai_model": None,
                "ai_warning": None,
                "evidence": ["operational_twin_state", "calibrated_model_risk"],
                "findings": [
                    f"Selected scenario: {payload.scenario}.",
                    f"Current cycle: {current['cycle']}.",
                    f"Current status: {status_label}.",
                    "No active incident is open.",
                ],
                "limitations": list(twin.limitations),
                "suggested_next_questions": [
                    "What is the current machine status?",
                    "Which signal is closest to a rule boundary?",
                    "Should I continue monitoring this scenario?",
                ],
            }
        response = copilot.investigate_live_incident(
            payload.question,
            package,
            scenario=payload.scenario,
            cycle=payload.cycle,
            conversation=[turn.model_dump() for turn in payload.conversation],
            mode=payload.mode,
        )
        evidence = response.evidence
        return {
            "scenario": payload.scenario,
            "cycle": payload.cycle,
            "answer": response.answer,
            "verified_answer": response.verified_answer,
            "ai_generated": response.ai_generated,
            "ai_status": response.ai_status,
            "ai_provider": get_settings().llm_provider if response.ai_generated else None,
            "ai_model": get_settings().groq_model if response.ai_generated else None,
            "ai_warning": response.ai_warning,
            "evidence": list(evidence.calculations_run),
            "findings": [finding.statement for finding in evidence.findings],
            "limitations": list(evidence.limitations),
            "suggested_next_questions": list(evidence.suggested_next_questions),
            "answer_mode": payload.mode,
            "planner_status": response.trace.planner_status,
            "grounding_status": response.trace.grounding_status,
            "investigation_trace": response.trace.model_dump(),
            "citations": [atom.model_dump() for atom in evidence.claim_ledger],
            "knowledge_sources": response.trace.knowledge_sources,
            "tool_round_count": response.trace.tool_round_count,
        }

    @app.post("/live/what-if", tags=["live operations"])
    def live_what_if(
        payload: LiveWhatIfRequest,
        copilot: Annotated[IndustrialCopilotService, Depends(_get_service)],
    ) -> dict[str, Any]:
        """Evaluate proposed controls through the real operational twin path."""

        snapshot = _live_snapshot(copilot, payload.scenario, payload.cycle)
        session = snapshot["session"]
        result = WhatIfEngine(risk_model=_live_risk_model()).evaluate(
            session,
            WhatIfInput(
                air_temperature_k=payload.air_temperature_k,
                process_temperature_k=payload.process_temperature_k,
                rotational_speed_rpm=payload.rotational_speed_rpm,
                torque_nm=payload.torque_nm,
                tool_wear_min=payload.tool_wear_min,
            ),
        )
        current = _serialize_twin(result.current_state)
        proposed = _serialize_twin(result.proposed_state)
        if snapshot["incident"] is not None:
            current["status"] = snapshot["incident"].severity
        summary = (
            f"Proposed outcome changes displayed status from {current['status']} to "
            f"{proposed['status']}; OSF margin changes from {current['osfMargin']:.0f} to "
            f"{proposed['osfMargin']:.0f} min Nm."
        )
        return {
            "scenario": payload.scenario,
            "cycle": payload.cycle,
            "current": current,
            "proposed": proposed,
            "summary": summary,
            "limitations": list(result.limitations),
        }

    return app


@lru_cache(maxsize=1)
def _live_risk_model():
    """Load the local calibrated model once for the live-console endpoints."""

    path = PROJECT_ROOT / "models" / f"{model_artifact_name('random_forest', 'engineering_augmented')}.joblib"
    return load_fitted_risk_model(path) if path.exists() else None


def _live_snapshot(copilot: IndustrialCopilotService, scenario: str, cycle: int) -> dict[str, Any]:
    events = _live_events(copilot, scenario)
    cursor = min(cycle, len(events))
    status = "idle" if cursor == 0 else "complete" if cursor == len(events) else "running"
    session = SimulationSession(
        session_id="API-LIVE-001",
        asset_id="MACHINE-01",
        source=events[0].source,
        source_label=events[0].source_label,
        status=status,
        speed=1,
        cursor_index=cursor,
        total_cycles=len(events),
        rolling_window_size=36,
        history=events[:cursor][-36:],
    )
    builder = OperationalTwinBuilder(risk_model=_live_risk_model())
    history_builder = OperationalTwinBuilder()
    incident_engine = IncidentEngine()
    incident = None
    for index in range(1, cursor + 1):
        replay_session = session.model_copy(
            update={"cursor_index": index, "status": "running", "history": events[:index][-36:]}
        )
        incident = incident_engine.evaluate(history_builder.build(replay_session)).incident
    twin = builder.build(session)
    incident = incident_engine.evaluate(twin).incident
    package = (
        build_incident_investigation_package(incident, twin, copilot.frame, recent_window_size=6, baseline_window_size=6, similar_k=8)
        if incident is not None and twin.current_telemetry is not None
        else None
    )
    return {"session": session, "twin": twin, "incident": incident, "package": package}


def _live_events(copilot: IndustrialCopilotService, scenario: str):
    if scenario == "OSF":
        return generate_osf_scenario(session_id="API-LIVE-001", cycles=36)
    if scenario == "HDF":
        return generate_hdf_scenario(session_id="API-LIVE-001", cycles=36)
    if scenario == "PWF":
        return generate_pwf_scenario(session_id="API-LIVE-001", cycles=36)
    replay = AI4IReplayEngine(copilot.frame, rolling_window_size=36)
    replay.start(speed=1)
    return tuple(event for _ in range(36) if (event := replay.next_event()) is not None)


def _serialize_live_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    twin = snapshot["twin"]
    incident = snapshot["incident"]
    package = snapshot["package"]
    result = _serialize_twin(twin)
    if incident is not None:
        result["status"] = incident.severity
    result["incident"] = None if incident is None else {
        "id": incident.incident_id,
        "severity": incident.severity,
        "opened": incident.first_cycle,
        "reason": incident.primary_reason,
        "evidence": list(incident.context.evidence),
    }
    result["changes"] = [] if package is None else [
        {
            "name": item.feature,
            "unit": _feature_unit(item.feature),
            "before": item.baseline_mean,
            "now": item.recent_mean,
            "percent": 0.0 if item.percent_change is None else item.percent_change * 100,
        }
        for item in package.what_changed.largest_changes
    ]
    result["similar"] = {"cases": [], "retrieved": 0, "failed": 0, "rate": 0.0, "topMode": "Not available"}
    if package is not None:
        similar = package.similar_historical_conditions
        result["similar"] = {
            "cases": [{"failed": bool(item.machine_failure)} for item in similar.observations],
            "retrieved": similar.returned_observation_count,
            "failed": similar.failed_observation_count,
            "rate": similar.similar_case_failure_rate or 0.0,
            "topMode": similar.most_common_failure_mode or "None",
        }
    return result


def _serialize_twin(twin) -> dict[str, Any]:
    margins = twin.rule_margins
    history = []
    builder = OperationalTwinBuilder()
    history_risks = _live_history_risks(twin.recent_history)
    for index, event in enumerate(twin.recent_history, start=1):
        state = builder.build(
            SimulationSession(
                session_id=twin.session_id,
                asset_id=twin.asset_id,
                source=event.source,
                source_label=event.source_label,
                status="running",
                speed=1,
                cursor_index=index,
                total_cycles=twin.total_cycles,
                rolling_window_size=36,
                history=(event,),
            )
        )
        serialized = _serialize_current(state)
        serialized["risk"] = history_risks[index - 1]
        history.append(serialized)
    return {
        **_serialize_current(twin),
        "cycle": twin.current_cycle,
        "total": twin.total_cycles,
        "stream": twin.simulation_status.upper(),
        "source": twin.source_label,
        "history": history,
        "margins": None if margins is None else margins.model_dump(),
        "risk_note": twin.risk_note,
    }


@lru_cache(maxsize=160)
def _live_history_risks(events: tuple[TelemetryEvent, ...]) -> tuple[float, ...]:
    """Score one visible telemetry history in a single cached model batch."""

    if not events:
        return ()
    model = _live_risk_model()
    if model is None:
        return tuple(0.0 for _ in events)
    operating = calculate_operating_features(telemetry_events_to_operating_frame(events))
    probabilities = model.estimator.predict_proba(operating.loc[:, model.input_features])[:, 1]
    return tuple(float(probability) for probability in probabilities)


def _feature_unit(feature: str) -> str:
    return {
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "Nm",
        "Tool wear [min]": "min",
        "Temperature delta [K]": "K",
        "Mechanical power [W]": "W",
        "Overstrain load [min Nm]": "min Nm",
    }.get(feature, "")


def _serialize_current(twin) -> dict[str, Any]:
    event = twin.current_telemetry
    engineered = twin.engineered
    return {
        "cycle": twin.current_cycle,
        "air": None if event is None else event.air_temperature_k,
        "process": None if event is None else event.process_temperature_k,
        "rpm": None if event is None else event.rotational_speed_rpm,
        "torque": None if event is None else event.torque_nm,
        "wear": None if event is None else event.tool_wear_min,
        "delta": None if engineered is None else engineered.temperature_delta_k,
        "power": None if engineered is None else engineered.mechanical_power_w,
        "load": None if engineered is None else engineered.overstrain_load_min_nm,
        "osfMargin": None if twin.rule_margins is None else twin.rule_margins.osf_remaining_margin_min_nm,
        "risk": 0.0 if twin.risk is None else twin.risk.failure_probability,
        "status": twin.machine_status,
    }


def _get_service(request: Request) -> IndustrialCopilotService:
    service = request.app.state.copilot_service
    if service is None:
        service = IndustrialCopilotService()
        request.app.state.copilot_service = service
    return service


def _run_tool(copilot: IndustrialCopilotService, call: ToolCall) -> dict[str, Any]:
    return _translate_errors(lambda: copilot.registry.execute(call))


def _model_risk(
    copilot: IndustrialCopilotService,
    model_name: str,
    feature_set: str,
    observation: PredictionInput,
) -> dict[str, Any]:
    return _run_tool(
        copilot,
        ToolCall(
            name="get_model_risk",
            arguments={
                "model_name": model_name,
                "feature_set": feature_set,
                "observation": observation.model_dump(),
            },
        ),
    )


def _translate_errors(operation: Callable[[], Result]) -> Result:
    try:
        return operation()
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (ToolArgumentError, UnknownToolError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


app = create_app()
