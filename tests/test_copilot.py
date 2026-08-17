"""Tests for the constrained offline copilot's planner, registry, and state handling."""

from __future__ import annotations

import pandas as pd
import pytest

from industrial_copilot.config import Settings
from industrial_copilot.copilot.planner import plan_offline_question
from industrial_copilot.copilot.schemas import ToolCall
from industrial_copilot.copilot.service import IndustrialCopilotService
from industrial_copilot.copilot.state import ConversationState
from industrial_copilot.llm.contracts import ExplanationResult, is_safe_explanation
from industrial_copilot.llm.gemini import GeminiEvidenceExplainer
from industrial_copilot.llm.groq import GroqEvidenceExplainer
from industrial_copilot.tools.industrial_tools import (
    ToolExecutionContext,
    build_industrial_registry,
)
from industrial_copilot.tools.registry import ToolArgumentError, UnknownToolError


def test_registry_rejects_unknown_tools_and_unvalidated_arguments(
    sample_ai4i_frame: pd.DataFrame, tmp_path: object
) -> None:
    registry = build_industrial_registry(ToolExecutionContext(sample_ai4i_frame, tmp_path))

    with pytest.raises(UnknownToolError):
        registry.execute(ToolCall(name="run_python", arguments={"code": "print(1)"}))
    with pytest.raises(ToolArgumentError):
        registry.execute(ToolCall(name="get_observation", arguments={"uid": "not-an-id"}))


def test_planner_retains_product_filter_for_a_follow_up() -> None:
    initial = plan_offline_question("What percentage failed?", ConversationState())
    state = ConversationState(previous_intent=initial.intent)
    filtered = plan_offline_question("Only L products.", state)
    assert filtered.intent == "failure_rate"
    assert filtered.state_updates is not None
    assert filtered.state_updates.product_types == ["L"]


def test_service_returns_evidence_and_uses_contextual_filters(sample_ai4i_frame: pd.DataFrame) -> None:
    service = IndustrialCopilotService(frame=sample_ai4i_frame, llm_enabled=False)
    first = service.ask("What percentage failed?")
    filtered = service.ask("Only L products.", ConversationState.model_validate(first.state))
    compared = service.ask("Now compare torque.", ConversationState.model_validate(filtered.state))

    assert first.evidence.metrics[1].value == "80.00%"
    assert filtered.evidence.filters.product_types == ["L"]
    assert compared.evidence.intent == "failed_healthy_comparison"
    assert compared.evidence.filters.product_types == ["L"]
    assert "compare_failed_vs_healthy" in compared.evidence.calculations_run


def test_unavailable_history_question_declines_without_inventing_results(
    sample_ai4i_frame: pd.DataFrame,
) -> None:
    response = IndustrialCopilotService(frame=sample_ai4i_frame, llm_enabled=False).ask(
        "What happened during the last 30 days?"
    )

    assert response.evidence.intent == "unavailable_data"
    assert response.evidence.calculations_run == []
    assert "does not contain timestamped history" in response.answer


def test_failure_investigation_has_only_whitelisted_tools() -> None:
    plan = plan_offline_question("Why are failures higher at high RPM?", ConversationState())

    assert plan.intent == "failure_investigation"
    assert {tool.name for tool in plan.tools} == {
        "failure_rate_by_range",
        "failure_mode_breakdown",
        "analyze_conditional_relationship",
        "discover_high_risk_regimes",
    }


class FakeExplainer:
    """Small deterministic stand-in for Gemini in a unit test."""

    def explain(self, _evidence: object) -> ExplanationResult:
        return ExplanationResult(status="generated", text="The evidence suggests a cautious comparison.")


def test_optional_gemini_explanation_never_replaces_the_deterministic_answer(
    sample_ai4i_frame: pd.DataFrame,
) -> None:
    response = IndustrialCopilotService(
        frame=sample_ai4i_frame,
        llm_enabled=True,
        llm_explainer=FakeExplainer(),
    ).ask("What percentage failed?")

    assert response.answer == "4 of 5 selected observations failed (80.00%)."
    assert response.llm_status == "generated"
    assert response.llm_explanation == "The evidence suggests a cautious comparison."


def test_blank_gemini_key_falls_back_without_importing_or_calling_a_provider(
    sample_ai4i_frame: pd.DataFrame,
) -> None:
    evidence = IndustrialCopilotService(
        frame=sample_ai4i_frame, llm_enabled=False
    ).ask("What percentage failed?").evidence
    result = GeminiEvidenceExplainer(
        Settings(llm_enabled=True, gemini_api_key=None)
    ).explain(evidence)

    assert result.status == "missing_api_key"
    assert result.text is None


def test_blank_groq_key_falls_back_without_importing_or_calling_a_provider(
    sample_ai4i_frame: pd.DataFrame,
) -> None:
    evidence = IndustrialCopilotService(
        frame=sample_ai4i_frame, llm_enabled=False
    ).ask("What percentage failed?").evidence
    result = GroqEvidenceExplainer(
        Settings(llm_provider="groq", llm_enabled=True, groq_api_key=None)
    ).explain(evidence)

    assert result.status == "missing_api_key"
    assert result.text is None


def test_llm_guard_rejects_incomplete_or_numerically_inventive_prose() -> None:
    assert is_safe_explanation("The current evidence supports checking tool condition first.")
    assert not is_safe_explanation("The current evidence supports checking")
    assert not is_safe_explanation("The risk is 99 percent.")
    assert not is_safe_explanation("Similar cases preceded a failure.")
    assert not is_safe_explanation("Similar cases ended with a failure.")
    assert not is_safe_explanation("This condition will fail.")
