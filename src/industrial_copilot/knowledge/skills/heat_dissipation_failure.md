---
title: AI4I Heat Dissipation Failure
authority: dataset_rule
source: UCI AI4I 2020 Predictive Maintenance Dataset — https://doi.org/10.24432/C5HS5C
failure_modes: HDF
signals: Air temperature, Process temperature, Rotational speed, Temperature delta
---
## Dataset mechanism
The synthetic HDF mechanism depends on the process-to-air temperature difference together with low rotational speed. Exact conditions and margins are calculated in backend engineering code.

## Investigation guidance
Review temperature delta and rotational speed together, compare the recent window to baseline, and inspect the relevant rule margins. Do not infer a real cooling-system defect from AI4I alone.

## Production caveat
Real thermal diagnosis needs cooling context, load, ambient effects, sensor health and genuine timestamps.
