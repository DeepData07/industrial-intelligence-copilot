"""Make one small optional-provider explanation request after local key configuration."""

from __future__ import annotations

import sys

from industrial_copilot.config import get_settings
from industrial_copilot.copilot.service import IndustrialCopilotService


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    api_key = settings.groq_api_key if settings.llm_provider == "groq" else settings.gemini_api_key
    if not settings.llm_enabled or not api_key:
        print(
            f"SKIPPED: Set LLM_ENABLED=true and the {settings.llm_provider.upper()} API key in .env "
            "before the live smoke test."
        )
        return

    response = IndustrialCopilotService().ask("What percentage failed?")
    model = settings.groq_model if settings.llm_provider == "groq" else settings.gemini_model
    print(f"Provider: {settings.llm_provider}")
    print(f"Model: {model}")
    print(f"Deterministic answer: {response.answer}")
    print(f"LLM status: {response.llm_status}")
    print(f"LLM explanation: {response.llm_explanation or response.llm_warning}")
    if response.llm_status != "generated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
