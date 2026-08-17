"""Dataset summary, subset failure rate, numeric range, and observation retrieval tools."""

from __future__ import annotations

import math

import pandas as pd

from industrial_copilot.analytics.common import (
    apply_filters,
    failure_rate,
    prepare_analysis_frame,
    resolve_filters,
    summary_statistics,
)
from industrial_copilot.analytics.models import (
    AnalysisFilters,
    DatasetSummary,
    FailureRateResult,
    NumericRange,
    ObservationRecord,
    RangeAnalysis,
    RangeFailureRate,
)
from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
    UID,
)
from industrial_copilot.features.engineering import (
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
    TEMPERATURE_DELTA,
)

SUMMARY_NUMERIC_COLUMNS = (
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    ROTATIONAL_SPEED,
    TORQUE,
    TOOL_WEAR,
    TEMPERATURE_DELTA,
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
)


def get_dataset_summary(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> DatasetSummary:
    """Summarize a selected operating population and its deterministic failure prevalence."""

    resolved = resolve_filters(filters)
    selected = apply_filters(prepare_analysis_frame(frame), resolved)
    observation_count, failed_count, rate = failure_rate(selected)
    return DatasetSummary(
        observation_count=observation_count,
        failed_observation_count=failed_count,
        failure_rate=rate,
        product_type_counts={
            product_type: int(count)
            for product_type, count in selected[PRODUCT_TYPE].value_counts().sort_index().items()
        },
        numeric_summary={
            column: summary_statistics(selected[column]) for column in SUMMARY_NUMERIC_COLUMNS
        },
        filters=resolved,
    )


def get_failure_rate(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> FailureRateResult:
    """Calculate the empirical machine-failure rate for an explicitly filtered population."""

    resolved = resolve_filters(filters)
    selected = apply_filters(prepare_analysis_frame(frame), resolved)
    observation_count, failed_count, rate = failure_rate(selected)
    return FailureRateResult(
        observation_count=observation_count,
        failed_observation_count=failed_count,
        failure_rate=rate,
        filters=resolved,
    )


def failure_rate_by_range(
    frame: pd.DataFrame,
    variable: str,
    ranges: list[NumericRange],
    filters: AnalysisFilters | None = None,
) -> RangeAnalysis:
    """Calculate failure rates in caller-specified numeric ranges without hidden binning."""

    if not ranges:
        raise ValueError("At least one explicit range is required.")
    if any(item.column != variable for item in ranges):
        raise ValueError("Every requested range must use the analysis variable.")

    resolved = resolve_filters(filters)
    selected = apply_filters(prepare_analysis_frame(frame), resolved)
    if variable not in selected.columns:
        raise ValueError(f"Unknown analysis variable: {variable}")
    if not pd.api.types.is_numeric_dtype(selected[variable]):
        raise ValueError(f"Analysis variable must be numeric: {variable}")

    results: list[RangeFailureRate] = []
    for requested_range in ranges:
        range_mask = pd.Series(True, index=selected.index)
        if requested_range.minimum is not None:
            range_mask &= selected[variable].ge(requested_range.minimum)
        if requested_range.maximum is not None:
            range_mask &= selected[variable].le(requested_range.maximum)
        subgroup = selected.loc[range_mask]
        count, failed_count, rate = failure_rate(subgroup)
        results.append(
            RangeFailureRate(
                label=_range_label(requested_range),
                minimum=requested_range.minimum,
                maximum=requested_range.maximum,
                observation_count=count,
                failed_observation_count=failed_count,
                failure_rate=rate if count else None,
            )
        )
    return RangeAnalysis(variable=variable, ranges=results, filters=resolved)


def get_observation(frame: pd.DataFrame, uid: int) -> ObservationRecord:
    """Retrieve one UID-keyed operating observation with calculated engineering features."""

    engineered = prepare_analysis_frame(frame)
    matches = engineered.loc[engineered[UID].eq(uid)]
    if matches.empty:
        raise KeyError(f"No observation found for UID {uid}.")
    if len(matches) > 1:
        raise ValueError(f"UID {uid} is not unique; resolve the data-contract error first.")

    row = matches.iloc[0]
    values = {column: _json_scalar(value) for column, value in row.items()}
    return ObservationRecord(uid=int(row[UID]), values=values)


def _range_label(requested_range: NumericRange) -> str:
    if requested_range.minimum is None:
        return f"≤ {requested_range.maximum:g}"
    if requested_range.maximum is None:
        return f"≥ {requested_range.minimum:g}"
    return f"{requested_range.minimum:g} to {requested_range.maximum:g}"


def _json_scalar(value: object) -> bool | float | int | str | None:
    """Convert pandas/NumPy scalar values to Pydantic/JSON-compatible primitives."""

    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bool, float, int, str)):
        return value
    return str(value)
