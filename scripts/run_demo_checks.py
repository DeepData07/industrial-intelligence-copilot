"""Run a compact, deterministic analysis against the local raw AI4I dataset."""

from __future__ import annotations

from industrial_copilot.analytics.comparisons import (
    compare_failed_vs_healthy,
    compare_product_types,
)
from industrial_copilot.analytics.descriptive import (
    failure_rate_by_range,
    get_dataset_summary,
)
from industrial_copilot.analytics.failure_analysis import failure_mode_breakdown
from industrial_copilot.analytics.models import AnalysisFilters, NumericRange
from industrial_copilot.analytics.similarity import find_similar_conditions
from industrial_copilot.data.loader import load_ai4i_data
from industrial_copilot.data.schema import ROTATIONAL_SPEED, TORQUE, UID


def main() -> None:
    """Print a concise JSON demonstration without writing data or model artifacts."""

    frame = load_ai4i_data()
    output = {
        "dataset_summary": get_dataset_summary(frame).model_dump(),
        "rpm_failure_rates": failure_rate_by_range(
            frame,
            variable=ROTATIONAL_SPEED,
            ranges=[
                NumericRange(column=ROTATIONAL_SPEED, maximum=1379),
                NumericRange(column=ROTATIONAL_SPEED, minimum=1380, maximum=1600),
                NumericRange(column=ROTATIONAL_SPEED, minimum=1601),
            ],
        ).model_dump(),
        "failed_vs_healthy_torque": compare_failed_vs_healthy(frame, [TORQUE]).model_dump(),
        "failure_modes": failure_mode_breakdown(frame).model_dump(),
        "product_types": compare_product_types(frame).model_dump(),
        "similar_conditions": find_similar_conditions(
            frame,
            uid=int(frame[UID].iloc[0]),
            k=3,
            filters=AnalysisFilters(),
        ).model_dump(),
    }
    import json

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
