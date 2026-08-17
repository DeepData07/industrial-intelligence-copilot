"""Render the deterministic Plotly charts into one local HTML preview."""

from __future__ import annotations

from pathlib import Path

import plotly.io as pio

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
from industrial_copilot.analytics.similarity import find_similar_conditions
from industrial_copilot.data.loader import load_ai4i_data
from industrial_copilot.data.schema import UID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "stage3_chart_preview.html"


def main() -> None:
    """Write a self-contained, local preview without modifying source data."""

    frame = load_ai4i_data()
    similarity = find_similar_conditions(frame, uid=int(frame[UID].iloc[0]), k=5)
    figures = [
        failure_rate_by_rpm_bands_chart(frame),
        failed_vs_healthy_torque_chart(frame),
        failed_vs_healthy_tool_wear_chart(frame),
        failure_mode_frequency_chart(frame),
        rpm_torque_failure_map_chart(frame),
        hdf_operating_envelope_chart(frame),
        torque_tool_wear_risk_heatmap_chart(frame),
        mechanical_power_operating_map_chart(frame),
        product_type_failure_comparison_chart(frame),
        similar_condition_outcome_chart(similarity),
    ]
    html_charts = [
        pio.to_html(figure, full_html=False, include_plotlyjs="cdn" if index == 0 else False)
        for index, figure in enumerate(figures)
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "<html><head><title>Industrial Intelligence Copilot — Chart Preview</title></head>"
        "<body style='font-family:Arial,sans-serif;margin:24px'>"
        "<h1>Industrial Intelligence Copilot — Chart Preview</h1>"
        + "<hr>".join(html_charts)
        + "</body></html>",
        encoding="utf-8",
    )
    print(f"Wrote chart preview: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
