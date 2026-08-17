---
title: AI4I Overstrain Failure
authority: dataset_rule
source: UCI AI4I 2020 Predictive Maintenance Dataset — https://doi.org/10.24432/C5HS5C
failure_modes: OSF
signals: Torque, Tool wear, Type, Overstrain load
---
## Dataset mechanism
AI4I represents overstrain through the interaction of torque and tool wear. The product-type threshold and current margin must be calculated by deterministic application code, not by the language model.

## Investigation guidance
Review the calculated wear-times-torque load, product-type threshold, remaining margin, recent torque/tool-wear changes, calibrated risk, and similar AI4I observations. A shrinking margin is exposure to the dataset-defined condition, not proof of a physical root cause.

## Limitation
AI4I contains no maintenance history, vibration or acoustic stream. Similar rows are cross-sectional reference observations, not a time sequence.
