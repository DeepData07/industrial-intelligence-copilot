# Low-Level Design

## System layout

```text
Browser (Vercel React UI)
        │ HTTPS
        ▼
FastAPI backend (Render)
        │
        ├── Scenario engine and operational twin
        ├── Engineering feature calculations and rule checks
        ├── Calibrated ML risk model
        ├── AI4I similar-condition retrieval
        ├── Evidence and grounding validator
        └── Together AI explanation request
                │
                └── Verified fallback if unavailable or invalid
```

## Frontend

The React/Vite frontend owns user interaction and presentation:

- scenario selection, playback speed and reset;
- telemetry cards, charts and machine-twin visual;
- incident display, questions and evidence trace;
- What-if sliders and maintenance-action interaction.

The frontend sends requests only to the FastAPI backend through `VITE_API_BASE_URL`. It contains no AI-provider secret.

## Backend

FastAPI exposes the operational APIs.

| Endpoint area | Purpose |
| --- | --- |
| `/health` | Service status and configured provider name |
| `/live/state` | Current scenario state, history, incident, changes and similar cases |
| `/live/copilot` | Evidence-first incident investigation and optional AI explanation |
| `/live/what-if` | Recalculate a proposed operating state |
| analysis/prediction endpoints | Dataset, statistical and model evidence |

## Operational twin and rules

The scenario engine creates ordered telemetry events. The twin builder converts events into the current operational state. Engineering calculations derive temperature delta, mechanical power and overstrain measures. The incident engine evaluates documented rule margins and calibrated-risk thresholds.

## Machine-learning layer

Persisted calibrated model artifacts estimate failure risk from raw and engineering-augmented AI4I features. The displayed percentage is a calibrated estimate from benchmark data, not a direct sensor measurement or a causal claim.

## Evidence layer

Before an explanation is requested, the backend builds an evidence package containing current state, findings, tool results, changes, similar cases, limitations and citations. Deterministic backend calculations remain the authority for facts.

## AI layer

Together AI is called from the Render backend using `TOGETHER_API_KEY`. The request uses `openai/gpt-oss-120b` through Together's OpenAI-compatible chat API and asks for structured JSON. The response passes two gates:

1. **Structural validation** — the response must be usable structured data.
2. **Evidence validation** — claims and numbers must map to supplied evidence atoms.

If either gate fails, or the provider is unavailable, the application returns the verified deterministic answer. The user can see which path produced the response from the UI badge.

## Security and deployment

- `TOGETHER_API_KEY` exists only in Render and an optional local `.env` file.
- Vercel has only the public backend URL in `VITE_API_BASE_URL`.
- `.env` is excluded from Git.
- GitHub contains the blank `.env.example`, source code, data and model artifacts needed to run the project.
