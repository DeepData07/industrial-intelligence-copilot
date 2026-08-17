"""Incident investigation helpers: what changed and historical similarity evidence."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.analytics.common import prepare_analysis_frame
from industrial_copilot.analytics.models import FeatureDifference, SimilarObservation
from industrial_copilot.analytics.similarity import SIMILARITY_FEATURES
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
    calculate_operating_features,
)
from industrial_copilot.simulation.incidents import Incident
from industrial_copilot.simulation.schemas import TelemetryEvent
from industrial_copilot.simulation.state import (
    OperationalTwinState,
    telemetry_events_to_operating_frame,
)

CHANGE_FEATURES = (
    ROTATIONAL_SPEED,
    TORQUE,
    TOOL_WEAR,
    TEMPERATURE_DELTA,
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
)


class FeatureChange(BaseModel):
    """Mean change from baseline window to recent window for one engineered signal."""

    model_config = ConfigDict(frozen=True)

    feature: str
    baseline_mean: float
    recent_mean: float
    absolute_change: float
    percent_change: float | None
    direction: str


class WhatChangedResult(BaseModel):
    """Deterministic recent-versus-baseline comparison for live telemetry."""

    model_config = ConfigDict(frozen=True)

    baseline_observation_count: int = Field(ge=0)
    recent_observation_count: int = Field(ge=0)
    changes: tuple[FeatureChange, ...]
    largest_changes: tuple[FeatureChange, ...]
    summary: str
    limitations: tuple[str, ...] = ()


class HistoricalSimilaritySummary(BaseModel):
    """Nearest historical AI4I evidence for a live operating point."""

    model_config = ConfigDict(frozen=True)

    candidate_count: int = Field(ge=0)
    returned_observation_count: int = Field(ge=0)
    failed_observation_count: int = Field(ge=0)
    similar_case_failure_rate: float | None = Field(default=None, ge=0, le=1)
    most_common_failure_mode: str | None = None
    observations: tuple[SimilarObservation, ...]
    note: str


class IncidentInvestigationPackage(BaseModel):
    """Compact package combining incident, change, and historical-memory evidence."""

    model_config = ConfigDict(frozen=True)

    incident_id: str
    asset_id: str
    session_id: str
    what_changed: WhatChangedResult
    similar_historical_conditions: HistoricalSimilaritySummary
    limitations: tuple[str, ...]


def calculate_what_changed(
    history: tuple[TelemetryEvent, ...],
    *,
    recent_window_size: int = 6,
    baseline_window_size: int = 6,
    top_n: int = 4,
) -> WhatChangedResult:
    """Compare recent live telemetry against the immediately preceding baseline window."""

    if recent_window_size < 1 or baseline_window_size < 1:
        raise ValueError("Window sizes must be at least 1.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    required = recent_window_size + baseline_window_size
    if len(history) < required:
        return WhatChangedResult(
            baseline_observation_count=max(0, len(history) - recent_window_size),
            recent_observation_count=min(len(history), recent_window_size),
            changes=(),
            largest_changes=(),
            summary=f"Need at least {required} telemetry events to compare recent and baseline windows.",
            limitations=("Insufficient telemetry history for deterministic change comparison.",),
        )

    baseline_events = history[-required:-recent_window_size]
    recent_events = history[-recent_window_size:]
    baseline = calculate_operating_features(telemetry_events_to_operating_frame(baseline_events))
    recent = calculate_operating_features(telemetry_events_to_operating_frame(recent_events))
    changes = tuple(
        _feature_change(feature, baseline[feature].mean(), recent[feature].mean())
        for feature in CHANGE_FEATURES
    )
    largest = tuple(
        sorted(
            changes,
            key=lambda item: abs(
                item.percent_change if item.percent_change is not None else item.absolute_change
            ),
            reverse=True,
        )[:top_n]
    )
    return WhatChangedResult(
        baseline_observation_count=len(baseline_events),
        recent_observation_count=len(recent_events),
        changes=changes,
        largest_changes=largest,
        summary=_change_summary(largest),
        limitations=("Change comparison uses simulated telemetry windows, not real plant history.",),
    )


def find_similar_historical_conditions_for_event(
    historical_frame: pd.DataFrame,
    event: TelemetryEvent,
    *,
    k: int = 10,
) -> HistoricalSimilaritySummary:
    """Find AI4I historical rows nearest to the current live operating point."""

    if k < 1:
        raise ValueError("k must be at least 1.")
    candidates = prepare_analysis_frame(historical_frame)
    if candidates.empty:
        return HistoricalSimilaritySummary(
            candidate_count=0,
            returned_observation_count=0,
            failed_observation_count=0,
            similar_case_failure_rate=None,
            observations=(),
            note="No historical candidates were available.",
        )

    target = calculate_operating_features(telemetry_events_to_operating_frame((event,))).iloc[0]
    candidate_features = candidates.loc[:, list(SIMILARITY_FEATURES)].astype(float)
    target_features = target.loc[list(SIMILARITY_FEATURES)].astype(float)
    means = candidate_features.mean()
    standard_deviations = candidate_features.std(ddof=0).replace(0, 1.0)
    standardized_candidates = (candidate_features - means) / standard_deviations
    standardized_target = (target_features - means) / standard_deviations
    distances = np.sqrt(((standardized_candidates - standardized_target) ** 2).sum(axis=1))

    nearest = candidates.assign(_distance=distances).nsmallest(k, "_distance")
    observations = tuple(
        _similar_observation(row, target_features, standard_deviations)
        for _, row in nearest.iterrows()
    )
    failed_count = sum(item.machine_failure for item in observations)
    mode_counts = Counter(mode for item in observations for mode in item.active_failure_modes)
    return HistoricalSimilaritySummary(
        candidate_count=len(candidates),
        returned_observation_count=len(observations),
        failed_observation_count=failed_count,
        similar_case_failure_rate=failed_count / len(observations) if observations else None,
        most_common_failure_mode=mode_counts.most_common(1)[0][0] if mode_counts else None,
        observations=observations,
        note=(
            "Historical similarity uses standardized engineered AI4I features. It is associative "
            "evidence, not proof that the live condition will fail."
        ),
    )


def build_incident_investigation_package(
    incident: Incident,
    twin_state: OperationalTwinState,
    historical_frame: pd.DataFrame,
    *,
    recent_window_size: int = 6,
    baseline_window_size: int = 6,
    similar_k: int = 10,
) -> IncidentInvestigationPackage:
    """Combine active incident context with deterministic change and similarity evidence."""

    if twin_state.current_telemetry is None:
        raise ValueError("Cannot investigate an incident without current telemetry.")
    what_changed = calculate_what_changed(
        twin_state.recent_history,
        recent_window_size=recent_window_size,
        baseline_window_size=baseline_window_size,
    )
    similar = find_similar_historical_conditions_for_event(
        historical_frame,
        twin_state.current_telemetry,
        k=similar_k,
    )
    return IncidentInvestigationPackage(
        incident_id=incident.incident_id,
        asset_id=incident.asset_id,
        session_id=incident.session_id,
        what_changed=what_changed,
        similar_historical_conditions=similar,
        limitations=(
            *incident.context.limitations,
            "Similar cases are AI4I historical observations, not real timestamped plant events.",
        ),
    )


def _feature_change(feature: str, baseline_mean: float, recent_mean: float) -> FeatureChange:
    absolute_change = float(recent_mean - baseline_mean)
    percent_change = None if abs(float(baseline_mean)) < 1e-12 else absolute_change / float(baseline_mean)
    if absolute_change > 0:
        direction = "increased"
    elif absolute_change < 0:
        direction = "decreased"
    else:
        direction = "unchanged"
    return FeatureChange(
        feature=feature,
        baseline_mean=float(baseline_mean),
        recent_mean=float(recent_mean),
        absolute_change=absolute_change,
        percent_change=percent_change,
        direction=direction,
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
        differences.append(
            FeatureDifference(
                feature=feature,
                target_value=target_value,
                similar_value=candidate_value,
                absolute_standardized_difference=(
                    abs(target_value - candidate_value) / float(standard_deviations[feature])
                ),
            )
        )
    return SimilarObservation(
        uid=int(candidate[UID]),
        distance=float(candidate["_distance"]),
        machine_failure=int(candidate[MACHINE_FAILURE]),
        active_failure_modes=[mode for mode in FAILURE_MODES if candidate[mode] == 1],
        important_differences=sorted(
            differences,
            key=lambda item: item.absolute_standardized_difference,
            reverse=True,
        )[:2],
    )


def _change_summary(largest_changes: tuple[FeatureChange, ...]) -> str:
    if not largest_changes:
        return "No comparable change evidence is available yet."
    lead = largest_changes[0]
    if lead.percent_change is None:
        return f"Largest recent change: {lead.feature} {lead.direction} by {lead.absolute_change:.2f}."
    return f"Largest recent change: {lead.feature} {lead.direction} by {lead.percent_change:.1%}."
