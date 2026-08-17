"""Interpretable two-condition risk-regime discovery with held-out confirmation."""

from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.model_selection import train_test_split

from industrial_copilot.analytics.common import (
    apply_filters,
    prepare_analysis_frame,
    resolve_filters,
)
from industrial_copilot.analytics.models import AnalysisFilters
from industrial_copilot.data.schema import (
    FAILURE_MODES,
    MACHINE_FAILURE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
)
from industrial_copilot.features.engineering import (
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
    TEMPERATURE_DELTA,
)
from industrial_copilot.statistics.associations import benjamini_hochberg, binary_association
from industrial_copilot.statistics.models import (
    RegimeCondition,
    RegimePartitionEvidence,
    RiskRegime,
    RiskRegimeDiscoveryResult,
)

DEFAULT_REGIME_FEATURES = (
    TORQUE,
    TOOL_WEAR,
    ROTATIONAL_SPEED,
    TEMPERATURE_DELTA,
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
)


def discover_high_risk_regimes(
    frame: pd.DataFrame,
    filters: AnalysisFilters | None = None,
    features: tuple[str, ...] = DEFAULT_REGIME_FEATURES,
    quantiles: tuple[float, ...] = (0.6, 0.75, 0.9),
    minimum_discovery_support: int = 100,
    minimum_confirmation_support: int = 40,
    minimum_risk_ratio: float = 1.5,
    fdr_alpha: float = 0.05,
    confirmation_fraction: float = 0.3,
    random_state: int = 42,
    max_regimes: int = 10,
) -> RiskRegimeDiscoveryResult:
    """Mine readable two-condition regimes on discovery data and test them on holdout data.

    This is exploratory discovery, not causal inference. Candidate hypotheses are generated
    from numeric tails, corrected with Benjamini-Hochberg on discovery data, and only then
    checked against an untouched stratified confirmation subset.
    """

    _validate_discovery_parameters(
        features,
        quantiles,
        minimum_discovery_support,
        minimum_confirmation_support,
        minimum_risk_ratio,
        fdr_alpha,
        confirmation_fraction,
        max_regimes,
    )
    resolved = resolve_filters(filters)
    selected = apply_filters(prepare_analysis_frame(frame), resolved)
    if len(selected) < minimum_discovery_support + minimum_confirmation_support:
        raise ValueError("Selected data is too small for the requested discovery and confirmation support.")
    _require_numeric_features(selected, features)
    discovery, confirmation = _split_discovery_confirmation(
        selected, confirmation_fraction, random_state
    )
    conditions = _candidate_conditions(discovery, features, quantiles)
    candidates: list[tuple[tuple[RegimeCondition, RegimeCondition], RegimePartitionEvidence]] = []
    for condition_pair in combinations(conditions, 2):
        if condition_pair[0].feature == condition_pair[1].feature:
            continue
        evidence = _evaluate_regime(discovery, condition_pair)
        if (
            evidence.observation_count >= minimum_discovery_support
            and evidence.failure_count >= 5
            and evidence.risk_ratio is not None
            and evidence.risk_ratio >= minimum_risk_ratio
            and evidence.fisher_exact_p_value is not None
        ):
            candidates.append((condition_pair, evidence))

    adjusted = benjamini_hochberg(
        [(_conditions_identifier(pair), evidence.fisher_exact_p_value) for pair, evidence in candidates],
        alpha=fdr_alpha,
    )
    adjusted_by_identifier = {item.identifier: item for item in adjusted}
    selected_regimes: list[RiskRegime] = []
    for pair, discovery_evidence in candidates:
        identifier = _conditions_identifier(pair)
        correction = adjusted_by_identifier[identifier]
        if not correction.rejected:
            continue
        confirmation_evidence = _evaluate_regime(confirmation, pair)
        stable = _is_confirmed(confirmation_evidence, minimum_confirmation_support)
        status = (
            "CONFIRMED"
            if stable
            else "INSUFFICIENT_DATA"
            if confirmation_evidence.observation_count < minimum_confirmation_support
            else "NOT_STABLE"
        )
        selected_regimes.append(
            RiskRegime(
                conditions=list(pair),
                status=status,
                discovery=discovery_evidence,
                confirmation=confirmation_evidence,
                discovery_q_value=correction.q_value,
                discovery_fdr_rejected=True,
                confirmation_stable=stable,
            )
        )
    selected_regimes.sort(
        key=lambda regime: (
            regime.status != "CONFIRMED",
            regime.discovery_q_value if regime.discovery_q_value is not None else 1.0,
            -(regime.discovery.risk_ratio or 0),
        )
    )
    return RiskRegimeDiscoveryResult(
        discovery_observation_count=len(discovery),
        confirmation_observation_count=len(confirmation),
        tested_regime_count=len(candidates),
        minimum_discovery_support=minimum_discovery_support,
        minimum_confirmation_support=minimum_confirmation_support,
        fdr_alpha=fdr_alpha,
        regimes=selected_regimes[:max_regimes],
        filters=resolved,
        note=(
            "Regimes were searched on discovery data, Benjamini-Hochberg corrected, and then "
            "evaluated on a held-out confirmation split. They are associations, not causal claims."
        ),
    )


def _validate_discovery_parameters(
    features: tuple[str, ...],
    quantiles: tuple[float, ...],
    minimum_discovery_support: int,
    minimum_confirmation_support: int,
    minimum_risk_ratio: float,
    fdr_alpha: float,
    confirmation_fraction: float,
    max_regimes: int,
) -> None:
    if len(features) < 2:
        raise ValueError("At least two features are required to mine interacting regimes.")
    if not quantiles or any(not 0 < value < 1 for value in quantiles):
        raise ValueError("Each quantile must be strictly between 0 and 1.")
    if minimum_discovery_support < 1 or minimum_confirmation_support < 1:
        raise ValueError("Minimum support values must be positive.")
    if minimum_risk_ratio <= 1:
        raise ValueError("minimum_risk_ratio must exceed 1.")
    if not 0 < fdr_alpha < 1 or not 0 < confirmation_fraction < 1:
        raise ValueError("fdr_alpha and confirmation_fraction must be between 0 and 1.")
    if max_regimes < 1:
        raise ValueError("max_regimes must be positive.")


def _split_discovery_confirmation(
    frame: pd.DataFrame, confirmation_fraction: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome = frame[MACHINE_FAILURE]
    stratify = outcome if outcome.nunique() == 2 and outcome.value_counts().min() >= 2 else None
    discovery, confirmation = train_test_split(
        frame,
        test_size=confirmation_fraction,
        random_state=random_state,
        stratify=stratify,
    )
    return discovery.copy(), confirmation.copy()


def _candidate_conditions(
    frame: pd.DataFrame, features: tuple[str, ...], quantiles: tuple[float, ...]
) -> list[RegimeCondition]:
    conditions: list[RegimeCondition] = []
    for feature in features:
        values = frame[feature]
        for quantile in quantiles:
            low_threshold = float(values.quantile(1 - quantile))
            high_threshold = float(values.quantile(quantile))
            conditions.extend(
                [
                    RegimeCondition(feature=feature, operator="<=", threshold=low_threshold),
                    RegimeCondition(feature=feature, operator=">=", threshold=high_threshold),
                ]
            )
    return _deduplicate_conditions(conditions)


def _deduplicate_conditions(conditions: list[RegimeCondition]) -> list[RegimeCondition]:
    unique: dict[tuple[str, str, float], RegimeCondition] = {}
    for condition in conditions:
        key = (condition.feature, condition.operator, round(condition.threshold, 12))
        unique[key] = condition
    return list(unique.values())


def _evaluate_regime(
    frame: pd.DataFrame, conditions: tuple[RegimeCondition, RegimeCondition]) -> RegimePartitionEvidence:
    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        if condition.operator == ">=":
            mask &= frame[condition.feature].ge(condition.threshold)
        else:
            mask &= frame[condition.feature].le(condition.threshold)
    association = binary_association(mask, frame[MACHINE_FAILURE])
    dominant_mode = _dominant_failure_mode(frame.loc[mask])
    failure_rate = association.exposed_failure_rate
    baseline_rate = float(frame[MACHINE_FAILURE].mean()) if len(frame) else None
    risk_lift = failure_rate - baseline_rate if failure_rate is not None and baseline_rate is not None else None
    return RegimePartitionEvidence(
        observation_count=association.exposed_observation_count,
        failure_count=association.exposed_failure_count,
        failure_rate=failure_rate,
        baseline_failure_rate=baseline_rate,
        risk_ratio=association.risk_ratio,
        risk_lift=risk_lift,
        odds_ratio=association.odds_ratio,
        odds_ratio_ci=association.odds_ratio_ci,
        fisher_exact_p_value=association.fisher_exact_p_value,
        dominant_failure_mode=dominant_mode,
    )


def _dominant_failure_mode(regime: pd.DataFrame) -> str | None:
    if regime.empty:
        return None
    counts = regime[list(FAILURE_MODES)].sum()
    if counts.max() == 0:
        return None
    return str(counts.idxmax())


def _is_confirmed(evidence: RegimePartitionEvidence, minimum_support: int) -> bool:
    interval = evidence.odds_ratio_ci
    return bool(
        evidence.observation_count >= minimum_support
        and evidence.risk_ratio is not None
        and evidence.risk_ratio > 1
        and interval.lower is not None
        and interval.lower > 1
        and evidence.fisher_exact_p_value is not None
        and evidence.fisher_exact_p_value < 0.05
    )


def _conditions_identifier(conditions: tuple[RegimeCondition, RegimeCondition]) -> str:
    return " AND ".join(condition.label for condition in conditions)


def _require_numeric_features(frame: pd.DataFrame, features: tuple[str, ...]) -> None:
    for feature in features:
        if feature not in frame.columns:
            raise ValueError(f"Unknown regime feature: {feature}")
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            raise ValueError(f"Regime feature must be numeric: {feature}")
