"""Optional Together AI adapter for evidence-grounded prose explanations."""

from __future__ import annotations

from industrial_copilot.config import Settings, get_settings
from industrial_copilot.copilot.schemas import EvidencePackage
from industrial_copilot.llm.contracts import (
    ExplanationResult,
    explanation_prompt,
    is_safe_explanation,
)


class TogetherEvidenceExplainer:
    """Call Together only after deterministic tools have built the EvidencePackage."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def explain(self, evidence: EvidencePackage) -> ExplanationResult:
        if not self.settings.llm_enabled:
            return ExplanationResult(status="disabled", warning="LLM_ENABLED is false.")
        if not self.settings.together_api_key:
            return ExplanationResult(
                status="missing_api_key",
                warning="TOGETHER_API_KEY is blank; using the deterministic offline answer.",
            )
        try:
            from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

            client = OpenAI(
                api_key=self.settings.together_api_key,
                base_url="https://api.together.ai/v1",
                timeout=self.settings.together_timeout_seconds,
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=self.settings.together_model,
                messages=[
                    {"role": "system", "content": "Answer only from supplied verified industrial evidence."},
                    {"role": "user", "content": explanation_prompt(evidence)},
                ],
                temperature=0.0,
                max_completion_tokens=300,
            )
            text = (response.choices[0].message.content or "").strip()
        except ImportError:
            return ExplanationResult(
                status="provider_error",
                warning="The openai package is unavailable; using the deterministic offline answer.",
            )
        except (APIConnectionError, APIError, APITimeoutError, OSError, TypeError, ValueError):
            return ExplanationResult(
                status="provider_error",
                warning="Together was unavailable; using the deterministic offline answer.",
            )
        if not is_safe_explanation(text):
            return ExplanationResult(
                status="invalid_output",
                warning="Together output did not meet the evidence-only response rules.",
            )
        return ExplanationResult(status="generated", text=text)
