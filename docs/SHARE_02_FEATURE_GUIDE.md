# Feature Guide

## Live operating scenarios

Four controlled scenarios demonstrate different operating conditions.

| Scenario | Demonstrates | Main rule context |
| --- | --- | --- |
| OSF | Overstrain condition | Torque × tool wear |
| HDF | Heat-dissipation condition | Temperature difference and speed |
| PWF | Mechanical-power condition | Power operating boundary |
| AI4I Replay | Dataset-based replay | Original AI4I-style observations |

Each scenario is disclosed as simulated or dataset-derived. It is not a live PLC connection.

## Live operating condition

The top cards summarise the latest machine state:

- **Rotational speed (rpm):** spindle rotation speed.
- **Torque (Nm):** turning force at the spindle.
- **Tool wear (min):** accumulated tool-use time used by the benchmark.
- **Temperature delta (K):** process temperature minus air temperature.
- **Mechanical power (kW):** calculated from torque and speed.
- **Failure risk (%):** calibrated model estimate, not a measured physical value.

## Telemetry stream

The chart presents the selected signals across simulation cycles. Each signal has its own normalised lane so trends remain readable even though their units differ.

## Active incident

An incident is opened when rule margins or calibrated risk cross the stated policy thresholds. The panel shows:

- the incident ID and opening cycle;
- the condition that triggered it;
- current risk and rule margin; and
- suggested inspection checks.

## What changed

The system compares the most recent five cycles with an earlier baseline window. It ranks the largest changes, helping focus the investigation on meaningful movement rather than a single reading.

## Similar historical conditions

The system retrieves nearby AI4I operating points using RPM, torque and tool wear. It reports how many were retrieved, how many had failure labels and the most common associated failure flag. Similarity is supporting evidence, not a prediction that the live asset will fail.

## AI Incident Copilot

The Copilot answers questions using the active scenario, current incident, recent-vs-baseline comparison and similar-case evidence.

- **Together AI** indicates a validated AI explanation generated from supplied evidence.
- **Verified fallback** indicates the deterministic evidence answer was used because the AI provider was unavailable or its response was rejected.
- **Quick** is intended for concise operational questions.
- **Deep** exposes a more detailed investigation trace and should be used selectively.

The Copilot is restricted from inventing values, causal conclusions, remaining useful life, unobserved machine history or machine commands.

## What-if analysis

Users can adjust proposed operating inputs such as speed, torque, wear and temperatures. Selecting **Recalculate proposed outcome** evaluates that proposed state using the same documented simulation and rule logic. This is decision support only; it does not control a machine.

## Suggested next checks and maintenance draft

Recommended checks can be marked complete. A maintenance-action draft records the suggested follow-up locally in the demonstration; it is not sent to a CMMS and does not issue a work order.
