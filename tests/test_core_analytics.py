from __future__ import annotations

import pytest

from industrial_copilot.analytics.comparisons import (
    compare_failed_vs_healthy,
    compare_product_types,
)
from industrial_copilot.analytics.descriptive import (
    failure_rate_by_range,
    get_dataset_summary,
    get_failure_rate,
    get_observation,
)
from industrial_copilot.analytics.failure_analysis import failure_mode_breakdown
from industrial_copilot.analytics.models import AnalysisFilters, NumericRange
from industrial_copilot.analytics.similarity import find_similar_conditions
from industrial_copilot.data.schema import MACHINE_FAILURE, ROTATIONAL_SPEED, TORQUE
from industrial_copilot.features.engineering import TEMPERATURE_DELTA


def test_summary_failure_rate_and_filters(sample_ai4i_frame) -> None:
    summary = get_dataset_summary(sample_ai4i_frame)
    l_only = get_failure_rate(sample_ai4i_frame, AnalysisFilters(product_types=["L"]))

    assert summary.observation_count == 5
    assert summary.failed_observation_count == 4
    assert summary.failure_rate == pytest.approx(0.8)
    assert summary.numeric_summary[TEMPERATURE_DELTA].mean == pytest.approx(9.6)
    assert l_only.observation_count == 3
    assert l_only.failed_observation_count == 2


def test_range_comparison_failed_healthy_and_product_types(sample_ai4i_frame) -> None:
    by_rpm = failure_rate_by_range(
        sample_ai4i_frame,
        ROTATIONAL_SPEED,
        [
            NumericRange(column=ROTATIONAL_SPEED, maximum=1380),
            NumericRange(column=ROTATIONAL_SPEED, minimum=1381),
        ],
    )
    comparison = compare_failed_vs_healthy(sample_ai4i_frame, [TORQUE])
    products = compare_product_types(sample_ai4i_frame)

    assert by_rpm.ranges[0].observation_count == 1
    assert by_rpm.ranges[0].failure_rate == 1.0
    assert by_rpm.ranges[1].failed_observation_count == 3
    assert comparison.variables[TORQUE]["healthy"].mean == 40.0
    assert comparison.variables[TORQUE]["failed"].mean == 35.0
    assert {group.product_type: group.observation_count for group in products.groups} == {
        "L": 3,
        "M": 1,
        "H": 1,
    }


def test_failure_modes_observation_lookup_and_similarity(sample_ai4i_frame) -> None:
    breakdown = failure_mode_breakdown(sample_ai4i_frame)
    observation = get_observation(sample_ai4i_frame, 4)
    similar = find_similar_conditions(sample_ai4i_frame, uid=1, k=2)

    modes = {item.failure_mode: item.flagged_observation_count for item in breakdown.modes}
    assert modes == {"TWF": 1, "HDF": 1, "PWF": 1, "OSF": 1, "RNF": 0}
    assert observation.uid == 4
    assert observation.values[MACHINE_FAILURE] == 1
    assert similar.candidate_count == 4
    assert len(similar.observations) == 2
    assert all(item.uid != 1 for item in similar.observations)
    assert similar.observations[0].distance <= similar.observations[1].distance


def test_invalid_analysis_inputs_are_rejected(sample_ai4i_frame) -> None:
    with pytest.raises(ValueError, match="Unknown filter variable"):
        get_failure_rate(
            sample_ai4i_frame,
            AnalysisFilters(numeric_ranges=[NumericRange(column="not_a_column", minimum=0)]),
        )
    with pytest.raises(ValueError, match="At least one explicit range"):
        failure_rate_by_range(sample_ai4i_frame, ROTATIONAL_SPEED, [])
    with pytest.raises(KeyError, match="UID 999"):
        get_observation(sample_ai4i_frame, 999)
    with pytest.raises(ValueError, match="k must be at least 1"):
        find_similar_conditions(sample_ai4i_frame, uid=1, k=0)
