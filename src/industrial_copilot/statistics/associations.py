"""Explicit 2×2 association effects and Benjamini-Hochberg correction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
from scipy.stats import fisher_exact, norm

from industrial_copilot.statistics.models import (
    AdjustedPValue,
    BinaryAssociation,
    ConfidenceInterval,
)


def binary_association(exposed: pd.Series, outcome: pd.Series) -> BinaryAssociation:
    """Calculate risk and odds measures from a boolean exposure and binary outcome.

    Zero cells use a Haldane-Anscombe 0.5 continuity correction for ratio confidence
    intervals. Raw counts and raw risks are always retained alongside corrected ratios.
    """

    if len(exposed) != len(outcome):
        raise ValueError("Exposure and outcome must have equal length.")
    valid = pd.DataFrame({"exposed": exposed, "outcome": outcome}).dropna()
    valid = valid.loc[valid["outcome"].isin([0, 1])]
    exposed_values = valid["exposed"].astype(bool)
    outcome_values = valid["outcome"].astype(int)
    a = int((exposed_values & outcome_values.eq(1)).sum())
    b = int((exposed_values & outcome_values.eq(0)).sum())
    c = int((~exposed_values & outcome_values.eq(1)).sum())
    d = int((~exposed_values & outcome_values.eq(0)).sum())
    exposed_total, unexposed_total = a + b, c + d
    exposed_risk = a / exposed_total if exposed_total else None
    unexposed_risk = c / unexposed_total if unexposed_total else None

    if not exposed_total or not unexposed_total:
        return BinaryAssociation(
            exposed_observation_count=exposed_total,
            exposed_failure_count=a,
            unexposed_observation_count=unexposed_total,
            unexposed_failure_count=c,
            exposed_failure_rate=exposed_risk,
            unexposed_failure_rate=unexposed_risk,
            risk_difference=None,
            risk_difference_ci=ConfidenceInterval(),
            risk_ratio=None,
            risk_ratio_ci=ConfidenceInterval(),
            odds_ratio=None,
            odds_ratio_ci=ConfidenceInterval(),
            fisher_exact_p_value=None,
            continuity_correction_applied=False,
        )

    risk_difference = exposed_risk - unexposed_risk
    risk_difference_se = math.sqrt(
        exposed_risk * (1 - exposed_risk) / exposed_total
        + unexposed_risk * (1 - unexposed_risk) / unexposed_total
    )
    z_score = norm.ppf(0.975)
    corrected = any(value == 0 for value in (a, b, c, d))
    ac, bc, cc, dc = (value + 0.5 for value in (a, b, c, d)) if corrected else (a, b, c, d)
    corrected_exposed_total, corrected_unexposed_total = ac + bc, cc + dc
    risk_ratio = (ac / corrected_exposed_total) / (cc / corrected_unexposed_total)
    risk_ratio_se = math.sqrt(1 / ac - 1 / corrected_exposed_total + 1 / cc - 1 / corrected_unexposed_total)
    odds_ratio = (ac * dc) / (bc * cc)
    odds_ratio_se = math.sqrt(1 / ac + 1 / bc + 1 / cc + 1 / dc)
    _, p_value = fisher_exact([[a, b], [c, d]])
    return BinaryAssociation(
        exposed_observation_count=exposed_total,
        exposed_failure_count=a,
        unexposed_observation_count=unexposed_total,
        unexposed_failure_count=c,
        exposed_failure_rate=exposed_risk,
        unexposed_failure_rate=unexposed_risk,
        risk_difference=risk_difference,
        risk_difference_ci=ConfidenceInterval(
            lower=risk_difference - z_score * risk_difference_se,
            upper=risk_difference + z_score * risk_difference_se,
        ),
        risk_ratio=risk_ratio,
        risk_ratio_ci=ConfidenceInterval(
            lower=math.exp(math.log(risk_ratio) - z_score * risk_ratio_se),
            upper=math.exp(math.log(risk_ratio) + z_score * risk_ratio_se),
        ),
        odds_ratio=odds_ratio,
        odds_ratio_ci=ConfidenceInterval(
            lower=math.exp(math.log(odds_ratio) - z_score * odds_ratio_se),
            upper=math.exp(math.log(odds_ratio) + z_score * odds_ratio_se),
        ),
        fisher_exact_p_value=float(p_value),
        continuity_correction_applied=corrected,
    )


def benjamini_hochberg(
    p_values: Sequence[tuple[str, float | None]], alpha: float = 0.05
) -> list[AdjustedPValue]:
    """Return FDR-adjusted q-values while preserving identifiers and missing p-values."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    valid = [(identifier, value) for identifier, value in p_values if value is not None]
    if any(not 0 <= value <= 1 for _, value in valid):
        raise ValueError("p-values must lie between 0 and 1.")
    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    adjusted_by_identifier: dict[str, float] = {}
    running_minimum = 1.0
    for reverse_rank, (identifier, p_value) in enumerate(reversed(ordered), start=1):
        rank = count - reverse_rank + 1
        running_minimum = min(running_minimum, p_value * count / rank)
        adjusted_by_identifier[identifier] = min(running_minimum, 1.0)
    return [
        AdjustedPValue(
            identifier=identifier,
            p_value=p_value,
            q_value=adjusted_by_identifier.get(identifier),
            rejected=(adjusted_by_identifier.get(identifier, 1.0) <= alpha),
        )
        for identifier, p_value in p_values
    ]
