from __future__ import annotations

import pandas as pd
import pytest

from industrial_copilot.data.schema import EXPECTED_COLUMNS, TOOL_WEAR, TORQUE
from industrial_copilot.statistics.associations import benjamini_hochberg, binary_association
from industrial_copilot.statistics.confounding import analyze_conditional_relationship
from industrial_copilot.statistics.risk_regimes import discover_high_risk_regimes


def _base_rows(outcomes: list[int], exposure: list[float], strata: list[str]) -> pd.DataFrame:
    rows = []
    for index, (outcome, exposure_value, stratum) in enumerate(zip(outcomes, exposure, strata), start=1):
        rows.append(
            [
                index,
                f"L{index:05d}",
                "L",
                300.0,
                310.0,
                1500,
                40.0,
                10,
                outcome,
                0,
                0,
                0,
                0,
                0,
            ]
        )
    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    frame["Exposure"] = exposure
    frame["Stratum"] = strata
    return frame


def _add_group(
    outcomes: list[int], exposure: list[float], strata: list[str], *, exposed_failures: int,
    exposed_healthy: int, unexposed_failures: int, unexposed_healthy: int, stratum: str
) -> None:
    outcomes.extend([1] * exposed_failures + [0] * exposed_healthy)
    exposure.extend([1.0] * (exposed_failures + exposed_healthy))
    strata.extend([stratum] * (exposed_failures + exposed_healthy))
    outcomes.extend([1] * unexposed_failures + [0] * unexposed_healthy)
    exposure.extend([0.0] * (unexposed_failures + unexposed_healthy))
    strata.extend([stratum] * (unexposed_failures + unexposed_healthy))


def test_binary_association_and_benjamini_hochberg() -> None:
    association = binary_association(
        pd.Series([True] * 100 + [False] * 100),
        pd.Series([1] * 20 + [0] * 80 + [1] * 10 + [0] * 90),
    )
    corrected = benjamini_hochberg([("a", 0.001), ("b", 0.02), ("c", 0.04), ("d", 0.2)])

    assert association.exposed_failure_rate == pytest.approx(0.2)
    assert association.unexposed_failure_rate == pytest.approx(0.1)
    assert association.risk_ratio == pytest.approx(2.0)
    assert association.odds_ratio == pytest.approx(2.25)
    assert association.continuity_correction_applied is False
    assert [item.q_value for item in corrected] == pytest.approx([0.004, 0.04, 0.053333333, 0.2])


def test_conditional_auditor_detects_supported_reversal() -> None:
    outcomes: list[int] = []
    exposure: list[float] = []
    strata: list[str] = []
    _add_group(
        outcomes,
        exposure,
        strata,
        exposed_failures=20,
        exposed_healthy=80,
        unexposed_failures=120,
        unexposed_healthy=280,
        stratum="low_baseline_risk",
    )
    _add_group(
        outcomes,
        exposure,
        strata,
        exposed_failures=400,
        exposed_healthy=400,
        unexposed_failures=60,
        unexposed_healthy=40,
        stratum="high_baseline_risk",
    )
    frame = _base_rows(outcomes, exposure, strata)

    result = analyze_conditional_relationship(
        frame,
        exposure="Exposure",
        exposure_threshold=0.5,
        conditioning_variable="Stratum",
        minimum_stratum_size=10,
    )

    assert result.aggregate_association.odds_ratio is not None
    assert result.aggregate_association.odds_ratio > 1
    assert result.mantel_haenszel.adjusted_odds_ratio is not None
    assert result.mantel_haenszel.adjusted_odds_ratio < 1
    assert result.effect_change == "CONFIRMED_REVERSAL"
    assert result.continuous_logistic_evidence.available is True


def test_regime_mining_uses_holdout_confirmation_without_mutating_source() -> None:
    outcomes: list[int] = []
    torque: list[float] = []
    wear: list[float] = []
    for index in range(600):
        high_risk_regime = index < 150
        torque.append(80.0 if high_risk_regime or index % 2 else 20.0)
        wear.append(200.0 if high_risk_regime or index % 3 else 10.0)
        outcomes.append(1 if (high_risk_regime and index < 210) or (not high_risk_regime and index < 220) else 0)
    frame = _base_rows(outcomes, torque, ["synthetic"] * len(outcomes))
    frame[TORQUE] = torque
    frame[TOOL_WEAR] = wear
    before = frame.copy(deep=True)

    result = discover_high_risk_regimes(
        frame,
        features=(TORQUE, TOOL_WEAR),
        quantiles=(0.75,),
        minimum_discovery_support=50,
        minimum_confirmation_support=20,
        minimum_risk_ratio=1.5,
        max_regimes=5,
    )

    assert result.discovery_observation_count + result.confirmation_observation_count == 600
    assert result.tested_regime_count > 0
    assert result.regimes
    assert any(regime.status == "CONFIRMED" for regime in result.regimes)
    pd.testing.assert_frame_equal(frame, before)


def test_conditional_auditor_rejects_invalid_input(sample_ai4i_frame) -> None:
    with pytest.raises(ValueError, match="must differ"):
        analyze_conditional_relationship(
            sample_ai4i_frame,
            exposure=TORQUE,
            exposure_threshold=40,
            conditioning_variable=TORQUE,
        )
    with pytest.raises(ValueError, match="Unknown exposure"):
        analyze_conditional_relationship(
            sample_ai4i_frame,
            exposure="missing",
            exposure_threshold=1,
            conditioning_variable="Type",
        )
