"""Failed-versus-healthy and product-type comparison tools."""

from __future__ import annotations

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
    FailedHealthyComparison,
    ProductTypeComparison,
    ProductTypeFailureRate,
)
from industrial_copilot.data.schema import MACHINE_FAILURE, PRODUCT_TYPE


def compare_failed_vs_healthy(
    frame: pd.DataFrame,
    variables: list[str],
    filters: AnalysisFilters | None = None,
) -> FailedHealthyComparison:
    """Compare selected numeric variables by observed machine-failure status."""

    if not variables:
        raise ValueError("At least one comparison variable is required.")
    resolved = resolve_filters(filters)
    selected = apply_filters(prepare_analysis_frame(frame), resolved)
    results = {}
    for variable in variables:
        if variable not in selected.columns:
            raise ValueError(f"Unknown comparison variable: {variable}")
        if not pd.api.types.is_numeric_dtype(selected[variable]):
            raise ValueError(f"Comparison variable must be numeric: {variable}")
        results[variable] = {
            "failed": summary_statistics(selected.loc[selected[MACHINE_FAILURE].eq(1), variable]),
            "healthy": summary_statistics(selected.loc[selected[MACHINE_FAILURE].eq(0), variable]),
        }
    return FailedHealthyComparison(variables=results, filters=resolved)


def compare_product_types(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> ProductTypeComparison:
    """Compare L, M, and H product groups under the same explicit remaining filters."""

    resolved = resolve_filters(filters)
    selected = apply_filters(prepare_analysis_frame(frame), resolved)
    groups: list[ProductTypeFailureRate] = []
    for product_type in ("L", "M", "H"):
        subgroup = selected.loc[selected[PRODUCT_TYPE].eq(product_type)]
        observation_count, failed_count, rate = failure_rate(subgroup)
        groups.append(
            ProductTypeFailureRate(
                product_type=product_type,
                observation_count=observation_count,
                failed_observation_count=failed_count,
                failure_rate=rate,
            )
        )
    return ProductTypeComparison(groups=groups, filters=resolved)
