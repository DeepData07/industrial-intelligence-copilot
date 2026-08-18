"""Safely verify Together AI configuration with one minimal model request."""

from __future__ import annotations

import json

from openai import OpenAI

from industrial_copilot.config import get_settings


def main() -> None:
    settings = get_settings()
    configured = bool(
        settings.llm_enabled
        and settings.llm_provider == "together"
        and settings.together_api_key
        and settings.together_api_key.strip()
    )
    print(f"Together enabled: {settings.llm_enabled}")
    print(f"Configured provider: {settings.llm_provider}")
    print(f"Together key configured: {bool(settings.together_api_key and settings.together_api_key.strip())}")
    print(f"Configured model: {settings.together_model}")
    if not configured:
        print("Result: NOT READY — set LLM_ENABLED=true, LLM_PROVIDER=together and TOGETHER_API_KEY in .env.")
        return

    try:
        client = OpenAI(
            api_key=settings.together_api_key,
            base_url="https://api.together.ai/v1",
            timeout=20.0,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=settings.together_model,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": 'Return exactly: {"status":"ok"}'},
            ],
            temperature=0.0,
            max_completion_tokens=100,
            response_format={"type": "json_object"},
        )
        json.loads(response.choices[0].message.content or "")
    except Exception as error:  # noqa: BLE001 - this script reports a provider diagnostic to the operator.
        print(f"Result: UNAVAILABLE — {type(error).__name__}: {error}")
        return

    print("Result: AVAILABLE — the key, account, selected model and network request all work.")


if __name__ == "__main__":
    main()
