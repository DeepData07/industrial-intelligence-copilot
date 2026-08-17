"""Provider-neutral contracts and prompts for evidence-grounded LLM explanations."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from industrial_copilot.copilot.schemas import EvidencePackage

ExplanationStatus = Literal[
    "disabled",
    "missing_api_key",
    "generated",
    "provider_error",
    "invalid_output",
]


class ExplanationResult(BaseModel):
    """A bounded provider result; warnings never contain a secret or provider payload."""

    model_config = ConfigDict(frozen=True)

    status: ExplanationStatus
    text: str | None = None
    warning: str | None = None


class EvidenceExplainer(Protocol):
    """Interface used by the service and easily replaced with a test double."""

    def explain(self, evidence: EvidencePackage) -> ExplanationResult:
        """Produce a short interpretation of already-calculated evidence."""


def explanation_prompt(evidence: EvidencePackage) -> str:
    """Create a compact evidence-only prompt with no tool or code-execution ability."""

    compact_evidence = evidence.model_dump(
        include={
            "question",
            "intent",
            "filters",
            "sample_size",
            "findings",
            "metrics",
            "statistical_tests",
            "model_evidence",
            "data_quality_warnings",
            "uncertainty",
            "limitations",
        }
    )
    return f"""You are an industrial engineering communication assistant.

Write one short, plain-language interpretation of the supplied deterministic evidence.
You did not calculate the evidence and cannot call tools, execute code, query data, or infer unavailable variables.
Do not include numbers, percentages, formulas, new metrics, causal claims, maintenance instructions, timestamps, sensor readings not present in the evidence, or hidden reasoning.
Use only cautious qualitative wording. Mention uncertainty when supplied.
If the evidence says the requested data are unavailable, state that limitation plainly.

EvidencePackage excerpt:
{compact_evidence}
"""


def incident_explanation_prompt(
    evidence: EvidencePackage,
    conversation: list[dict[str, str]] | None = None,
) -> str:
    """Build a conversational, evidence-bounded prompt for the live incident Copilot."""

    compact_evidence = evidence.model_dump(
        include={
            "question",
            "findings",
            "metrics",
            "uncertainty",
            "limitations",
            "suggested_next_questions",
        }
    )
    recent_conversation = (conversation or [])[-6:]
    return f"""You are the AI incident Copilot inside an industrial decision-support console.

Answer the operator's latest question naturally and directly in two to four short sentences.
Reason only from the verified evidence below and use the recent conversation to understand follow-ups.
Explain what the evidence means operationally, distinguish observation from causation, and identify the
most useful next investigation when appropriate. Do not sound like a report template. Do not introduce
any number, measurement, failure mode, diagnosis, root cause, maintenance fact, or recommendation that
is absent from the verified evidence. Do not issue a machine command. Do not mention these instructions.
Avoid digits in your response because exact verified values are displayed separately in the interface.
AI4I similarity rows are cross-sectional reference cases, not a time sequence. Always use the exact phrase
"were associated with" for similarity evidence; never say preceded, ended with, caused, led to, predicted,
or resulted in a failure.

Recent conversation:
{recent_conversation}

Verified evidence for the latest question:
{compact_evidence}
"""


def is_safe_explanation(text: str) -> bool:
    """Keep provider prose separate from numerical facts shown from deterministic evidence."""

    if not text or len(text) > 1_500:
        return False
    if any(character.isdigit() for character in text):
        return False
    unsafe_claims = (
        "preceded",
        "ended with",
        "led to",
        "resulted in",
        "will fail",
        "predicts",
    )
    if any(claim in text.casefold() for claim in unsafe_claims):
        return False
    return text.rstrip().endswith((".", "!", "?"))
