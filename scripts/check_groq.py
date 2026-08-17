"""Safely verify Groq configuration and a minimal model request.

Never prints an API key.  Run with the project virtual environment.
"""

from __future__ import annotations

import json

from groq import Groq

from industrial_copilot.config import get_settings


def main() -> None:
    settings = get_settings()
    configured = bool(settings.llm_enabled and settings.groq_api_key and settings.groq_api_key.strip())
    print(f"Groq enabled: {settings.llm_enabled}")
    print(f"Groq key configured: {bool(settings.groq_api_key and settings.groq_api_key.strip())}")
    print(f"Configured model: {settings.groq_model}")
    if not configured:
        print("Result: NOT READY — set LLM_ENABLED=true and GROQ_API_KEY in .env.")
        return

    try:
        response = Groq(api_key=settings.groq_api_key, timeout=12.0, max_retries=0).chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": 'Return exactly: {"status":"ok"}'},
            ],
            temperature=0.0,
            # gpt-oss can use part of a short budget for internal reasoning before emitting JSON.
            max_completion_tokens=100,
        )
        content = response.choices[0].message.content or ""
        json.loads(content)
    except Exception as error:  # noqa: BLE001 - this script reports a provider diagnostic to the operator.
        print(f"Result: UNAVAILABLE — {type(error).__name__}: {error}")
        return

    print("Result: AVAILABLE — the key, account, selected model and network request all work.")


if __name__ == "__main__":
    main()
