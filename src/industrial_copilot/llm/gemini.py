"""Small optional Gemini adapter for evidence-grounded prose explanations."""

from __future__ import annotations

from industrial_copilot.config import Settings, get_settings
from industrial_copilot.copilot.schemas import EvidencePackage
from industrial_copilot.llm.contracts import (
    ExplanationResult,
    explanation_prompt,
    is_safe_explanation,
)


class GeminiEvidenceExplainer:
    """Call Gemini only after deterministic tools have built the EvidencePackage."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def explain(self, evidence: EvidencePackage) -> ExplanationResult:
        """Return safe prose, or a status that lets the caller keep its offline answer."""

        if not self.settings.llm_enabled:
            return ExplanationResult(status="disabled", warning="LLM_ENABLED is false.")
        if not self.settings.gemini_api_key:
            return ExplanationResult(
                status="missing_api_key",
                warning="GEMINI_API_KEY is blank; using the deterministic offline answer.",
            )
        try:
            from google import genai
            from google.genai import errors, types

            client = genai.Client(api_key=self.settings.gemini_api_key)
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=explanation_prompt(evidence),
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=220),
            )
            text = (response.text or "").strip()
        except ImportError:
            return ExplanationResult(
                status="provider_error",
                warning="The google-genai package is unavailable; using the deterministic offline answer.",
            )
        except (errors.APIError, OSError, TypeError, ValueError):
            return ExplanationResult(
                status="provider_error",
                warning="Gemini was unavailable; using the deterministic offline answer.",
            )
        if not is_safe_explanation(text):
            return ExplanationResult(
                status="invalid_output",
                warning="Gemini output did not meet the evidence-only response rules.",
            )
        return ExplanationResult(status="generated", text=text)
