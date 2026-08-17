"""Conditional relationship audit with stratified and continuous adjusted evidence."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import StratifiedTable

from industrial_copilot.analytics.common import (
    apply_filters,
    prepare_analysis_frame,
    resolve_filters,
)
from industrial_copilot.analytics.models import AnalysisFilters
from industrial_copilot.data.schema import MACHINE_FAILURE
from industrial_copilot.statistics.associations import binary_association
from industrial_copilot.statistics.models import (
    BinaryAssociation,
    ConditionalRelationshipResult,
    ConfidenceInterval,
    ContinuousLogisticEvidence,
    MantelHaenszelEstimate,
    StratumAssociation,
)


def analyze_conditional_relationship(
    frame: pd.DataFrame,
    exposure: str,
    exposure_threshold: float,
    conditioning_variable: str,
    filters: AnalysisFilters | None = None,
    numeric_strata: int = 4,
    minimum_stratum_size: int = 30,
) -> ConditionalRelationshipResult:
    """Compare an exposure above a threshold before and after conditioning on a variable.

    Numeric conditioning variables use quantile strata only for transparent stratified
    reporting. A complementary logistic model keeps both the exposure and numeric controls
    continuous, standardized per one standard deviation.
    """

    if exposure == conditioning_variable:
        raise ValueError("Exposure and conditioning variable must differ.")
    if numeric_strata < 2:
        raise ValueError("numeric_strata must be at least 2.")
    if minimum_stratum_size < 1:
        raise ValueError("minimum_stratum_size must be at least 1.")

    resolved = resolve_filters(filters)
    selected = apply_filters(prepare_analysis_frame(frame), resolved)
    _require_numeric(selected, exposure)
    if conditioning_variable not in selected.columns:
        raise ValueError(f"Unknown conditioning variable: {conditioning_variable}")
    if selected.empty:
        raise ValueError("Conditional analysis has no observations after filtering.")

    exposed = selected[exposure].ge(exposure_threshold)
    aggregate = binary_association(exposed, selected[MACHINE_FAILURE])
    strata, method = _build_strata(selected, conditioning_variable, numeric_strata)
    stratum_results: list[StratumAssociation] = []
    contingency_tables: list[np.ndarray] = []
    for stratum_label, stratum_index in strata.items():
        stratum = selected.loc[stratum_index]
        if len(stratum) < minimum_stratum_size:
            continue
        stratum_exposed = stratum[exposure].ge(exposure_threshold)
        association = binary_association(stratum_exposed, stratum[MACHINE_FAILURE])
        stratum_results.append(StratumAssociation(stratum=stratum_label, association=association))
        table = _raw_table(association)
        if min(table.ravel()) > 0:
            contingency_tables.append(table)

    mantel_haenszel = _mantel_haenszel(contingency_tables)
    continuous = _continuous_logistic_evidence(selected, exposure, conditioning_variable)
    effect_change, interpretation = _classify_effect_change(aggregate, mantel_haenszel)
    return ConditionalRelationshipResult(
        exposure=exposure,
        exposure_threshold=exposure_threshold,
        conditioning_variable=conditioning_variable,
        conditioning_method=method,
        selected_observation_count=len(selected),
        aggregate_association=aggregate,
        stratum_associations=stratum_results,
        mantel_haenszel=mantel_haenszel,
        continuous_logistic_evidence=continuous,
        effect_change=effect_change,
        interpretation=interpretation,
        filters=resolved,
    )


def _build_strata(
    frame: pd.DataFrame, variable: str, numeric_strata: int
) -> tuple[dict[str, pd.Index], str]:
    values = frame[variable]
    if pd.api.types.is_numeric_dtype(values):
        if values.nunique(dropna=True) < 2:
            raise ValueError(f"Conditioning variable needs at least two distinct values: {variable}")
        bins = pd.qcut(values, q=numeric_strata, duplicates="drop")
        labels = bins.astype(str)
        method = f"quantile strata (up to {numeric_strata} groups)"
    else:
        labels = values.astype(str)
        method = "categorical strata"
    return {str(label): frame.index[labels.eq(label)] for label in sorted(labels.dropna().unique())}, method


def _raw_table(association: BinaryAssociation) -> np.ndarray:
    """Return exposed/unexposed by failed/healthy counts for statsmodels."""

    return np.array(
        [
            [
                association.exposed_failure_count,
                association.exposed_observation_count - association.exposed_failure_count,
            ],
            [
                association.unexposed_failure_count,
                association.unexposed_observation_count - association.unexposed_failure_count,
            ],
        ]
    )


def _mantel_haenszel(tables: list[np.ndarray]) -> MantelHaenszelEstimate:
    if len(tables) < 2:
        return MantelHaenszelEstimate(
            available=False,
            stratum_count=len(tables),
            confidence_interval=ConfidenceInterval(),
            note="At least two strata with both exposure groups and outcomes are required.",
        )
    try:
        stratified = StratifiedTable(np.stack(tables, axis=2))
        confidence_interval = stratified.oddsratio_pooled_confint()
        return MantelHaenszelEstimate(
            available=True,
            stratum_count=len(tables),
            adjusted_odds_ratio=float(stratified.oddsratio_pooled),
            confidence_interval=ConfidenceInterval(
                lower=float(confidence_interval[0]), upper=float(confidence_interval[1])
            ),
            null_odds_p_value=float(stratified.test_null_odds().pvalue),
            homogeneity_p_value=float(stratified.test_equal_odds().pvalue),
            note="Mantel-Haenszel common odds ratio across eligible strata.",
        )
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as error:
        return MantelHaenszelEstimate(
            available=False,
            stratum_count=len(tables),
            confidence_interval=ConfidenceInterval(),
            note=f"Mantel-Haenszel estimate unavailable: {error}",
        )


def _continuous_logistic_evidence(
    frame: pd.DataFrame, exposure: str, conditioning_variable: str
) -> ContinuousLogisticEvidence:
    outcome = frame[MACHINE_FAILURE].astype(int)
    if outcome.nunique() < 2:
        return _unavailable_continuous_evidence("Both outcome classes are required for logistic adjustment.")
    design = pd.DataFrame({"exposure": _standardize(frame[exposure])}, index=frame.index)
    if design["exposure"].isna().any() or design["exposure"].nunique() < 2:
        return _unavailable_continuous_evidence("Exposure has insufficient numeric variation.")
    adjustment_variables = [conditioning_variable]
    if pd.api.types.is_numeric_dtype(frame[conditioning_variable]):
        design[conditioning_variable] = _standardize(frame[conditioning_variable])
    else:
        encoded = pd.get_dummies(frame[conditioning_variable], prefix=conditioning_variable, drop_first=True)
        if encoded.empty:
            return _unavailable_continuous_evidence("Conditioning variable has only one category.")
        design = pd.concat([design, encoded.astype(float)], axis=1)
    try:
        fitted = sm.GLM(outcome, sm.add_constant(design, has_constant="add"), family=sm.families.Binomial()).fit()
        coefficient = float(fitted.params["exposure"])
        interval = fitted.conf_int().loc["exposure"]
        return ContinuousLogisticEvidence(
            available=True,
            exposure_odds_ratio_per_standard_deviation=math.exp(coefficient),
            confidence_interval=ConfidenceInterval(
                lower=math.exp(float(interval.iloc[0])), upper=math.exp(float(interval.iloc[1]))
            ),
            p_value=float(fitted.pvalues["exposure"]),
            adjustment_variables=adjustment_variables,
            note=(
                "Logistic association adjusted for the conditioning variable; exposure is "
                "standardized, so the odds ratio is per one exposure standard deviation."
            ),
        )
    except (np.linalg.LinAlgError, ValueError, sm.tools.sm_exceptions.PerfectSeparationError) as error:
        return _unavailable_continuous_evidence(f"Continuous logistic adjustment unavailable: {error}")


def _standardize(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=0)
    if standard_deviation == 0 or pd.isna(standard_deviation):
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / standard_deviation


def _unavailable_continuous_evidence(note: str) -> ContinuousLogisticEvidence:
    return ContinuousLogisticEvidence(
        available=False,
        confidence_interval=ConfidenceInterval(),
        adjustment_variables=[],
        note=note,
    )


def _classify_effect_change(
    aggregate: BinaryAssociation, adjusted: MantelHaenszelEstimate
) -> tuple[str, str]:
    if not adjusted.available or aggregate.odds_ratio is None or adjusted.adjusted_odds_ratio is None:
        return "INSUFFICIENT_SUPPORT", "Insufficient eligible strata for a stable adjusted comparison."
    aggregate_direction = _effect_direction(aggregate.odds_ratio)
    adjusted_direction = _effect_direction(adjusted.adjusted_odds_ratio)
    adjusted_significant = (
        adjusted.confidence_interval.lower is not None
        and adjusted.confidence_interval.upper is not None
        and not (adjusted.confidence_interval.lower <= 1 <= adjusted.confidence_interval.upper)
        and adjusted.null_odds_p_value is not None
        and adjusted.null_odds_p_value < 0.05
    )
    if aggregate_direction and adjusted_direction and aggregate_direction != adjusted_direction:
        if adjusted_significant:
            return (
                "CONFIRMED_REVERSAL",
                (
                    "The aggregate and Mantel-Haenszel adjusted odds-ratio directions differ; "
                    "this is a stratified association reversal, not proof of causation."
                ),
            )
        return (
            "INSUFFICIENT_SUPPORT",
            (
                "Aggregate and adjusted directions differ, but the adjusted estimate is not stable "
                "enough to label a reversal."
            ),
        )
    if abs(math.log(adjusted.adjusted_odds_ratio)) <= abs(math.log(aggregate.odds_ratio)) * 0.5:
        return (
            "RELATIONSHIP_WEAKENED",
            "The adjusted odds ratio is materially closer to 1 than the aggregate estimate.",
        )
    return (
        "NO_MEANINGFUL_CHANGE",
        "The adjusted odds ratio remains in the same direction with no material weakening criterion met.",
    )


def _effect_direction(odds_ratio: float) -> int:
    if odds_ratio > 1:
        return 1
    if odds_ratio < 1:
        return -1
    return 0


def _require_numeric(frame: pd.DataFrame, variable: str) -> None:
    if variable not in frame.columns:
        raise ValueError(f"Unknown exposure variable: {variable}")
    if not pd.api.types.is_numeric_dtype(frame[variable]):
        raise ValueError(f"Exposure variable must be numeric: {variable}")
