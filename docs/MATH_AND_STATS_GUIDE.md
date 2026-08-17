# Math and Statistics Guide

## Purpose

This project uses statistics to stress-test descriptive patterns. An association can guide an investigation, but it does not prove that changing one variable will cause a failure outcome to change.

## Binary exposure table

For an exposure (for example, `RPM >= 1600`) and machine failure:

| | Failed | Healthy |
|---|---:|---:|
| Exposed | a | b |
| Unexposed | c | d |

The code reports raw counts and these measures:

- Exposed risk: `a / (a + b)`
- Unexposed risk: `c / (c + d)`
- Risk difference: exposed risk − unexposed risk
- Risk ratio: exposed risk / unexposed risk
- Odds ratio: `(a × d) / (b × c)`

Risk-ratio and odds-ratio confidence intervals use the log scale. If any 2×2 cell is zero, the system applies the Haldane–Anscombe `+0.5` continuity correction only to ratio estimates and their intervals; raw counts and raw risks remain unchanged. Fisher's exact test supplies the two-sided p-value.

## Conditional relationship auditor

The auditor first reports the aggregate comparison. It then stratifies by a conditioning variable:

- Categorical variables such as product type use their natural categories.
- Numeric variables use up to four quantile strata for readable reporting.

Across eligible strata it calculates the Mantel–Haenszel common odds ratio and its confidence interval. It also fits a logistic association model that keeps numeric exposure/control variables continuous and standardizes them. The reported continuous odds ratio is therefore per one standard deviation of the exposure.

The result can be `CONFIRMED_REVERSAL` only when aggregate and adjusted odds-ratio directions differ and the adjusted estimate is statistically distinguishable from one. Otherwise the relationship is labelled weakened, unchanged, or insufficiently supported. This is not a causal conclusion and should not be called Simpson's paradox without these criteria.

## Hidden risk-regime miner

The miner searches readable two-condition tail rules such as:

```text
Torque >= threshold AND Tool wear >= threshold
```

It creates a reproducible stratified 70/30 discovery/confirmation split:

1. Search only the discovery partition.
2. Require a minimum subgroup support and at least five failures.
3. Calculate support, failure rate, baseline rate, risk ratio, risk lift, odds ratio, confidence interval, p-value, and dominant mode.
4. Apply Benjamini–Hochberg FDR correction across all tested candidates.
5. Evaluate surviving discoveries once on the untouched confirmation partition.

A regime is `CONFIRMED` only when confirmation support is adequate, its risk ratio remains above one, the odds-ratio confidence interval is above one, and Fisher's p-value is below 0.05. Otherwise it is `NOT_STABLE` or `INSUFFICIENT_DATA`.

## Important limits

- The AI4I data are synthetic, cross-sectional observations; `UID` is not a longitudinal machine history.
- Failure-mode flags may overlap and have a documented source-label disagreement with `Machine failure`; no labels are changed.
- Statistical significance is not operational importance or causality.
- Threshold searches are exploratory, even after holdout confirmation. Plant deployment needs genuine time ordering, machine-specific baselines, and pre-registered validation.
