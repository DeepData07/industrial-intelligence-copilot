"""Select the configured optional explanation provider without changing deterministic logic."""

from __future__ import annotations

from industrial_copilot.config import Settings
from industrial_copilot.llm.contracts import EvidenceExplainer
from industrial_copilot.llm.gemini import GeminiEvidenceExplainer
from industrial_copilot.llm.groq import GroqEvidenceExplainer
from industrial_copilot.llm.together import TogetherEvidenceExplainer


def build_explainer(settings: Settings) -> EvidenceExplainer:
    """Return the one configured provider; all callers retain deterministic fallback."""

    if settings.llm_provider == "groq":
        return GroqEvidenceExplainer(settings)
    if settings.llm_provider == "together":
        return TogetherEvidenceExplainer(settings)
    return GeminiEvidenceExplainer(settings)
