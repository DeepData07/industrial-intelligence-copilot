"""Run compact conditional-effect and held-out risk-regime demonstrations."""

from __future__ import annotations

import json

from industrial_copilot.data.loader import load_ai4i_data
from industrial_copilot.data.schema import ROTATIONAL_SPEED, TORQUE
from industrial_copilot.statistics.confounding import analyze_conditional_relationship
from industrial_copilot.statistics.risk_regimes import discover_high_risk_regimes


def main() -> None:
    """Print structured evidence without modifying data or fitting predictive models."""

    frame = load_ai4i_data()
    conditional = analyze_conditional_relationship(
        frame,
        exposure=ROTATIONAL_SPEED,
        exposure_threshold=1600,
        conditioning_variable=TORQUE,
    )
    regimes = discover_high_risk_regimes(frame)
    print(
        json.dumps(
            {
                "rpm_conditioned_on_torque": conditional.model_dump(),
                "hidden_risk_regimes": regimes.model_dump(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
