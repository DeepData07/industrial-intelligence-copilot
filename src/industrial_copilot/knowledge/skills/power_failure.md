---
title: AI4I Power Failure
authority: dataset_rule
source: UCI AI4I 2020 Predictive Maintenance Dataset — https://doi.org/10.24432/C5HS5C
failure_modes: PWF
signals: Rotational speed, Torque, Mechanical power
---
## Dataset mechanism
Mechanical power depends on torque and angular velocity. The AI4I documented power condition is evaluated by deterministic code; the language model must never calculate or invent power.

## Investigation guidance
Inspect RPM, torque, calculated mechanical power, distance to the documented boundary, recent changes, model risk and similar observations. Do not equate mechanical power with unobserved electrical consumption or energy cost.
