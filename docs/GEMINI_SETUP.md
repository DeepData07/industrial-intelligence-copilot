# Gemini setup (optional provider)

The copilot works without Gemini. Gemini is only used to add a short, non-numerical explanation after deterministic tools have already calculated the evidence.

## Create a key

1. Open [Google AI Studio](https://aistudio.google.com/).
2. Sign in, accept the applicable terms, and choose **Get API key**.
3. Create or select a Google Cloud project, then choose **Create API key**.
4. Copy the key once. Treat it as a password: do not paste it into chat, source code, Git, screenshots, or a frontend.

Google documents this API-key flow and key restrictions in its [API key guide](https://ai.google.dev/gemini-api/docs/api-key). `gemini-2.5-flash` remains a stable supported text model for this optional low-latency explanation use case according to the [model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash).

## Configure locally

Create `.env` from `.env.example`, then set only these local values:

```text
LLM_ENABLED=true
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

`.env` is git-ignored. Keep the repository example blank:

```text
GEMINI_API_KEY=
```

## Verify

```powershell
.\.venv\Scripts\python.exe scripts\run_gemini_smoke_test.py
```

Expected successful output includes `Gemini status: generated`, the deterministic failure-rate answer, and a short Gemini interpretation without numerical calculations. If the key is blank, disabled, invalid, rate-limited, or Gemini is unavailable, the script reports the condition and the core copilot continues to work offline.
