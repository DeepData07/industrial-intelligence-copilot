"""Explainable nearest-neighbour retrieval over standardized engineering variables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_copilot.analytics.common import (
    apply_filters,
    prepare_analysis_frame,
    resolve_filters,
)
from industrial_copilot.analytics.models import (
    AnalysisFilters,
    FeatureDifference,
    SimilarConditionsResult,
    SimilarObservation,
)
from industrial_copilot.data.schema import (
    FAILURE_MODES,
    MACHINE_FAILURE,
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

SIMILARITY_FEATURES = (
    TEMPERATURE_DELTA,
    ROTATIONAL_SPEED,
    TORQUE,
    TOOL_WEAR,
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
)


def find_similar_conditions(
    frame: pd.DataFrame,
    uid: int,
    k: int = 5,
    filters: AnalysisFilters | None = None,
) -> SimilarConditionsResult:
    """Find up to k nearest historical operating conditions using standardized features.

    Standardization statistics are fit on eligible candidates only. The target UID is excluded,
    preventing a distance-zero observation from appearing as its own historical neighbour.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")
    resolved = resolve_filters(filters)
    engineered = prepare_analysis_frame(frame)
    target_rows = engineered.loc[engineered[UID].eq(uid)]
    if target_rows.empty:
        raise KeyError(f"No observation found for UID {uid}.")
    if len(target_rows) > 1:
        raise ValueError(f"UID {uid} is not unique; resolve the data-contract error first.")
    target = target_rows.iloc[0]

    candidates = apply_filters(engineered, resolved)
    candidates = candidates.loc[candidates[UID].ne(uid)].copy()
    if candidates.empty:
        return SimilarConditionsResult(
            target_uid=uid,
            candidate_count=0,
            similar_case_failure_rate=None,
            feature_columns=list(SIMILARITY_FEATURES),
            observations=[],
            filters=resolved,
        )

    candidate_features = candidates.loc[:, list(SIMILARITY_FEATURES)].astype(float)
    target_features = target.loc[list(SIMILARITY_FEATURES)].astype(float)
    means = candidate_features.mean()
    standard_deviations = candidate_features.std(ddof=0).replace(0, 1.0)
    standardized_candidates = (candidate_features - means) / standard_deviations
    standardized_target = (target_features - means) / standard_deviations
    distances = np.sqrt(((standardized_candidates - standardized_target) ** 2).sum(axis=1))

    nearest = candidates.assign(_distance=distances).nsmallest(k, "_distance")
    observations = [
        _similar_observation(row, target_features, standard_deviations) for _, row in nearest.iterrows()
    ]
    similar_case_failure_rate = (
        sum(item.machine_failure for item in observations) / len(observations) if observations else None
    )
    return SimilarConditionsResult(
        target_uid=uid,
        candidate_count=len(candidates),
        similar_case_failure_rate=similar_case_failure_rate,
        feature_columns=list(SIMILARITY_FEATURES),
        observations=observations,
        filters=resolved,
    )


def _similar_observation(
    candidate: pd.Series,
    target_features: pd.Series,
    standard_deviations: pd.Series,
) -> SimilarObservation:
    differences = []
    for feature in SIMILARITY_FEATURES:
        target_value = float(target_features[feature])
        candidate_value = float(candidate[feature])
        standardized_difference = abs(target_value - candidate_value) / float(standard_deviations[feature])
        differences.append(
            FeatureDifference(
                feature=feature,
                target_value=target_value,
                similar_value=candidate_value,
                absolute_standardized_difference=standardized_difference,
            )
        )
    important_differences = sorted(
        differences, key=lambda item: item.absolute_standardized_difference, reverse=True
    )[:2]
    return SimilarObservation(
        uid=int(candidate[UID]),
        distance=float(candidate["_distance"]),
        machine_failure=int(candidate[MACHINE_FAILURE]),
        active_failure_modes=[mode for mode in FAILURE_MODES if candidate[mode] == 1],
        important_differences=important_differences,
    )
