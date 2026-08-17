# Groq setup (recommended optional provider)

The copilot works fully offline. Groq only writes a short qualitative explanation after the deterministic tools have created the EvidencePackage.

## Why Groq here

Groq provides a Free tier with published usage limits and no credit card required for signup; a payment method is required only to upgrade to its Developer tier. This makes it suitable for local evaluation. Check current per-account limits in the Groq Console because free quotas can change. [Groq free-tier FAQ](https://community.groq.com/t/is-there-a-free-tier-and-what-are-its-limits/790) · [rate limits](https://console.groq.com/docs/rate-limits)

The default model is `openai/gpt-oss-120b`. It has a 131K-token context window and supports structured outputs, so it exceeds the modest requirement here: a brief evidence-grounded explanation. The application sends no tools to Groq and does not permit model-driven tool use, browsing, or code execution. [Groq model documentation](https://console.groq.com/docs/model/openai/gpt-oss-120b)

## Create a key

1. Open [GroqCloud Console](https://console.groq.com/) and sign in using Google, GitHub, or email.
2. Open [API Keys](https://console.groq.com/keys).
3. Select **Create API Key**, give it a name such as `industrial-copilot-local`, and copy it.
4. Keep it private. Do not paste it into chat, source code, Git, or screenshots.

Groq’s official quickstart uses the same key flow and `GROQ_API_KEY` environment variable. [Quickstart](https://console.groq.com/docs/quickstart)

## Configure locally

Create `.env` from `.env.example` if it does not already exist. Then set:

```text
LLM_ENABLED=true
LLM_PROVIDER=groq
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
```

`.env` is git-ignored. Keep the repository example blank:

```text
GROQ_API_KEY=
```

## Verify

```powershell
.\.venv\Scripts\python.exe scripts\run_llm_smoke_test.py
```

Expected successful output includes `Provider: groq`, `LLM status: generated`, the deterministic failure-rate answer, and a short non-numerical explanation. If the key is blank, invalid, rate-limited, or Groq is unavailable, the deterministic copilot remains usable and the smoke test reports the provider condition.
