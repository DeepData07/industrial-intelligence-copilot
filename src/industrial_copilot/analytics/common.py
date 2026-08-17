"""Shared filtering and descriptive helpers for deterministic analytics."""

from __future__ import annotations

import math

import pandas as pd

from industrial_copilot.analytics.models import AnalysisFilters, SummaryStatistics
from industrial_copilot.data.schema import MACHINE_FAILURE, PRODUCT_TYPE
from industrial_copilot.features.engineering import calculate_engineering_features


def prepare_analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a feature-complete copy suitable for analysis, never mutating raw input."""

    return calculate_engineering_features(frame)


def apply_filters(frame: pd.DataFrame, filters: AnalysisFilters | None = None) -> pd.DataFrame:
    """Return a subset using only explicit, validated product, label, and numeric filters."""

    resolved = filters or AnalysisFilters()
    mask = pd.Series(True, index=frame.index)
    if resolved.product_types is not None:
        mask &= frame[PRODUCT_TYPE].isin(resolved.product_types)
    if resolved.machine_failure is not None:
        mask &= frame[MACHINE_FAILURE].eq(resolved.machine_failure)
    if resolved.failure_mode is not None:
        mask &= frame[resolved.failure_mode].eq(1)
    for numeric_range in resolved.numeric_ranges:
        if numeric_range.column not in frame.columns:
            raise ValueError(f"Unknown filter variable: {numeric_range.column}")
        if not pd.api.types.is_numeric_dtype(frame[numeric_range.column]):
            raise ValueError(f"Filter variable must be numeric: {numeric_range.column}")
        if numeric_range.minimum is not None:
            mask &= frame[numeric_range.column].ge(numeric_range.minimum)
        if numeric_range.maximum is not None:
            mask &= frame[numeric_range.column].le(numeric_range.maximum)
    return frame.loc[mask].copy(deep=True)


def resolve_filters(filters: AnalysisFilters | None) -> AnalysisFilters:
    """Replace an omitted filter with its explicit empty representation."""

    return filters or AnalysisFilters()


def summary_statistics(values: pd.Series) -> SummaryStatistics:
    """Calculate transparent numeric descriptive statistics without imputation."""

    valid = values.dropna()
    if valid.empty:
        return SummaryStatistics(count=0)
    standard_deviation = float(valid.std(ddof=1)) if len(valid) > 1 else None
    if standard_deviation is not None and math.isnan(standard_deviation):
        standard_deviation = None
    return SummaryStatistics(
        count=int(valid.count()),
        mean=float(valid.mean()),
        median=float(valid.median()),
        minimum=float(valid.min()),
        maximum=float(valid.max()),
        standard_deviation=standard_deviation,
    )


def failure_rate(frame: pd.DataFrame) -> tuple[int, int, float]:
    """Return selected rows, failures, and deterministic empirical failure rate."""

    observations = len(frame)
    failed = int(frame[MACHINE_FAILURE].eq(1).sum())
    return observations, failed, (failed / observations if observations else 0.0)
