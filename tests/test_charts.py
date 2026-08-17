from __future__ import annotations

import plotly.graph_objects as go
import pytest

from industrial_copilot.analytics.charts import (
    failed_vs_healthy_tool_wear_chart,
    failed_vs_healthy_torque_chart,
    failure_mode_frequency_chart,
    failure_rate_by_rpm_bands_chart,
    hdf_operating_envelope_chart,
    mechanical_power_operating_map_chart,
    product_type_failure_comparison_chart,
    rpm_torque_failure_map_chart,
    similar_condition_outcome_chart,
    torque_tool_wear_risk_heatmap_chart,
)
from industrial_copilot.analytics.models import AnalysisFilters, SimilarConditionsResult
from industrial_copilot.analytics.similarity import find_similar_conditions


@pytest.mark.parametrize(
    "chart_factory",
    [
        failure_rate_by_rpm_bands_chart,
        failed_vs_healthy_torque_chart,
        failed_vs_healthy_tool_wear_chart,
        failure_mode_frequency_chart,
        rpm_torque_failure_map_chart,
        hdf_operating_envelope_chart,
        torque_tool_wear_risk_heatmap_chart,
        mechanical_power_operating_map_chart,
        product_type_failure_comparison_chart,
    ],
)
def test_required_chart_factories_return_populated_plotly_figures(
    sample_ai4i_frame, chart_factory
) -> None:
    figure = chart_factory(sample_ai4i_frame)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) > 0
    assert figure.layout.title.text


def test_similar_condition_chart_uses_structured_similarity_result(sample_ai4i_frame) -> None:
    result = find_similar_conditions(sample_ai4i_frame, uid=1, k=2)
    figure = similar_condition_outcome_chart(result)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert figure.data[0].mode == "markers+text"
    assert figure.layout.yaxis.ticktext == ("Healthy", "Failed")


def test_chart_inputs_are_validated(sample_ai4i_frame) -> None:
    with pytest.raises(ValueError, match="Chart input is empty"):
        failure_rate_by_rpm_bands_chart(
            sample_ai4i_frame,
            AnalysisFilters(product_types=[]),
        )
    with pytest.raises(ValueError, match="Risk heatmap requires"):
        constant = sample_ai4i_frame.copy()
        constant["Torque [Nm]"] = 40.0
        torque_tool_wear_risk_heatmap_chart(constant)
    with pytest.raises(ValueError, match="Similar-condition chart requires"):
        result = SimilarConditionsResult(
            target_uid=1,
            candidate_count=0,
            similar_case_failure_rate=None,
            feature_columns=[],
            observations=[],
            filters=AnalysisFilters(),
        )
        similar_condition_outcome_chart(result)
