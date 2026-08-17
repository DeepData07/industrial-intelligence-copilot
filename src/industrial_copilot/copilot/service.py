"""Application service for safe, offline, deterministic AI4I copilot requests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from industrial_copilot.config import PROJECT_ROOT, get_settings
from industrial_copilot.copilot.agentic import AgenticIncidentResult, BoundedIncidentAgent
from industrial_copilot.copilot.answer import render_offline_answer
from industrial_copilot.copilot.evidence import build_evidence_package
from industrial_copilot.copilot.incident import answer_incident_question
from industrial_copilot.copilot.planner import plan_offline_question
from industrial_copilot.copilot.schemas import EvidencePackage, OfflineCopilotResponse
from industrial_copilot.copilot.state import ConversationState, updated_state
from industrial_copilot.data.loader import load_ai4i_data
from industrial_copilot.llm.contracts import EvidenceExplainer, ExplanationResult
from industrial_copilot.llm.factory import build_explainer
from industrial_copilot.simulation.investigation import IncidentInvestigationPackage
from industrial_copilot.tools.industrial_tools import (
    ToolExecutionContext,
    build_industrial_registry,
)


class IndustrialCopilotService:
    """Run a fixed, validated tool plan and return its source-grounded evidence."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        models_directory: Path | None = None,
        llm_enabled: bool | None = None,
        llm_explainer: EvidenceExplainer | None = None,
    ) -> None:
        settings = get_settings()
        self.frame = frame if frame is not None else load_ai4i_data(settings.raw_data_path)
        self.llm_enabled = settings.llm_enabled if llm_enabled is None else llm_enabled
        self.llm_explainer = llm_explainer or build_explainer(settings)
        self.registry = build_industrial_registry(
            ToolExecutionContext(self.frame, models_directory or PROJECT_ROOT / "models")
        )

    def ask(self, question: str, state: ConversationState | None = None) -> OfflineCopilotResponse:
        """Answer one supported request with whitelisted deterministic tools only."""

        prior_state = state or ConversationState()
        plan = plan_offline_question(question, prior_state)
        results = {call.name: self.registry.execute(call) for call in plan.tools}
        filters = plan.state_updates or prior_state.current_filters
        evidence = build_evidence_package(question, plan, results, filters)
        explanation = self._explain(evidence)
        next_state = updated_state(
            prior_state,
            plan.intent,
            filters,
            result=results or None,
            uid=_planned_uid(plan),
        )
        return OfflineCopilotResponse(
            answer=render_offline_answer(plan, evidence),
            evidence=evidence,
            state=next_state.model_dump(),
            llm_explanation=explanation.text,
            llm_status=explanation.status,
            llm_warning=explanation.warning,
        )

    def ask_about_incident(
        self,
        question: str,
        incident_package: IncidentInvestigationPackage,
        state: ConversationState | None = None,
        conversation: list[dict[str, str]] | None = None,
    ) -> OfflineCopilotResponse:
        """Answer a live incident follow-up using compact deterministic incident context."""

        prior_state = state or ConversationState()
        answer, evidence = answer_incident_question(question, incident_package)
        explanation = self._explain_incident(evidence, conversation)
        next_state = updated_state(
            prior_state,
            "incident_investigation",
            prior_state.current_filters,
            result=evidence.tool_results,
            incident_id=incident_package.incident_id,
        )
        return OfflineCopilotResponse(
            answer=answer,
            evidence=evidence,
            state=next_state.model_dump(),
            llm_explanation=explanation.text,
            llm_status=explanation.status,
            llm_warning=explanation.warning,
        )

    def investigate_live_incident(
        self,
        question: str,
        incident_package: IncidentInvestigationPackage,
        *,
        scenario: str,
        cycle: int,
        conversation: list[dict[str, str]] | None = None,
        mode: str = "quick",
        state: ConversationState | None = None,
    ) -> AgenticIncidentResult:
        """Run the bounded live agent; deterministic evidence remains the failure-safe path."""

        settings = get_settings().model_copy(update={"llm_enabled": self.llm_enabled})
        return BoundedIncidentAgent(settings=settings).investigate(
            question,
            incident_package,
            scenario=scenario,
            cycle=cycle,
            conversation=conversation,
            mode=mode,
            state=state,
        )

    def _explain(self, evidence: EvidencePackage) -> ExplanationResult:
        if not self.llm_enabled:
            return ExplanationResult(status="disabled", warning="LLM_ENABLED is false.")
        return self.llm_explainer.explain(evidence)

    def _explain_incident(
        self,
        evidence: EvidencePackage,
        conversation: list[dict[str, str]] | None,
    ) -> ExplanationResult:
        if not self.llm_enabled:
            return ExplanationResult(status="disabled", warning="LLM_ENABLED is false.")
        explain_incident = getattr(self.llm_explainer, "explain_incident", None)
        if callable(explain_incident):
            return explain_incident(evidence, conversation)
        return self.llm_explainer.explain(evidence)


def _planned_uid(plan: object) -> int | None:
    tools = getattr(plan, "tools", [])
    for call in tools:
        uid = call.arguments.get("uid")
        if isinstance(uid, int):
            return uid
    return None
