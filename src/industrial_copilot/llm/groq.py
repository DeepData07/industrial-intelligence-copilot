"""Optional Groq adapter for evidence-grounded prose explanations."""

from __future__ import annotations

from industrial_copilot.config import Settings, get_settings
from industrial_copilot.copilot.schemas import EvidencePackage
from industrial_copilot.llm.contracts import (
    ExplanationResult,
    explanation_prompt,
    incident_explanation_prompt,
    is_safe_explanation,
)


class GroqEvidenceExplainer:
    """Call Groq only after deterministic tools have built the EvidencePackage."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def explain(self, evidence: EvidencePackage) -> ExplanationResult:
        """Return safe prose, or a status that leaves the deterministic answer intact."""

        return self._generate(explanation_prompt(evidence))

    def explain_incident(
        self,
        evidence: EvidencePackage,
        conversation: list[dict[str, str]] | None = None,
    ) -> ExplanationResult:
        """Answer a live follow-up conversationally from its verified evidence package."""

        return self._generate(incident_explanation_prompt(evidence, conversation))

    def _generate(self, prompt: str) -> ExplanationResult:
        """Call Groq with a bounded latency and validate its prose before returning it."""

        if not self.settings.llm_enabled:
            return ExplanationResult(status="disabled", warning="LLM_ENABLED is false.")
        if not self.settings.groq_api_key:
            return ExplanationResult(
                status="missing_api_key",
                warning="GROQ_API_KEY is blank; using the deterministic offline answer.",
            )
        try:
            from groq import APIConnectionError, APIError, APITimeoutError, Groq

            client = Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.settings.groq_timeout_seconds,
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=self.settings.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer only from the supplied verified industrial evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                # Stable demo behavior: the evidence is fixed, so creative variation is undesirable.
                temperature=0.0,
                # Reasoning models count internal reasoning against this budget as well.
                max_completion_tokens=800,
            )
            text = (response.choices[0].message.content or "").strip()
        except ImportError:
            return ExplanationResult(
                status="provider_error",
                warning="The groq package is unavailable; using the deterministic offline answer.",
            )
        except (APIConnectionError, APIError, APITimeoutError, OSError, TypeError, ValueError):
            return ExplanationResult(
                status="provider_error",
                warning="Groq was unavailable; using the deterministic offline answer.",
            )
        if not is_safe_explanation(text):
            return ExplanationResult(
                status="invalid_output",
                warning="Groq output did not meet the evidence-only response rules.",
            )
        return ExplanationResult(status="generated", text=text)
