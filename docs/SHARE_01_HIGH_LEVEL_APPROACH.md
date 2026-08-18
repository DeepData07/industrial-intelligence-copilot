# From a Static Dataset to a Live Operational Copilot

## The starting point

The source data is the AI4I 2020 predictive-maintenance dataset. It contains individual machine observations: speed, torque, temperatures, tool wear and failure labels. It is a useful benchmark, but it is not a live plant feed and does not contain real timestamps or PLC signals.

The central question was therefore: how can a row-based maintenance dataset be experienced as an operational system without claiming that it is connected to a real factory?

## The approach

The solution turns verified AI4I observations and documented synthetic scenarios into an ordered sequence of machine cycles. This sequence acts as an **operational machine twin** for one milling asset, MACHINE-01.

At every cycle, the system calculates the current operating condition:

- rotational speed and torque;
- tool wear;
- air and process temperature difference;
- mechanical power;
- engineering rule margins; and
- calibrated machine-failure risk.

This lets a user watch the machine condition develop rather than inspect disconnected spreadsheet rows.

## The operational story

1. **Sense** — A selected scenario streams simulated telemetry cycle by cycle.
2. **Detect** — Rule margins and calibrated model risk are evaluated. A watch, warning or incident can be opened.
3. **Understand** — The current window is compared with its earlier baseline, and similar AI4I operating conditions are retrieved.
4. **Investigate** — The Copilot explains the verified evidence in natural language. It does not create the facts itself.
5. **Decide** — A What-if workspace lets the user propose different operating inputs and see a recalculated outcome.
6. **Act** — Suggested checks can be marked complete and converted into a local maintenance-action draft.

## Why the design is useful

The design keeps each responsibility clear:

| Responsibility | System component |
| --- | --- |
| Measurements and engineering calculations | Deterministic backend logic |
| Failure-risk estimate | Calibrated machine-learning model |
| Similar-condition evidence | AI4I nearest-condition retrieval |
| Natural-language explanation | Together AI, constrained by verified evidence |
| Safe operation when AI is unavailable | Verified deterministic fallback |

The result is not a claim of a deployed factory control system. It is a transparent, evidence-first demonstration of how historical maintenance data, engineering rules, machine learning and an AI explanation layer can work together in an industrial decision-support workflow.

## Final outcome

The application provides a realistic operational experience while preserving the limitations of the dataset. It supports investigation and discussion; it does not issue machine commands, estimate remaining useful life, claim causal proof, or replace an engineer’s decision.
