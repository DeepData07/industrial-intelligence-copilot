from __future__ import annotations

import pytest

from industrial_copilot.features.engineering import (
    HDF_RULE_CONDITION,
    MECHANICAL_POWER,
    OSF_RULE_CONDITION,
    OVERSTRAIN_LOAD,
    OVERSTRAIN_THRESHOLD,
    PWF_RULE_CONDITION,
    TEMPERATURE_DELTA,
    calculate_operating_features,
)
from industrial_copilot.simulation.scenarios import (
    SYNTHETIC_SCENARIO_LABEL,
    events_to_operating_frame,
    generate_hdf_scenario,
    generate_osf_scenario,
    generate_pwf_scenario,
    generate_scenario_events,
)


def test_osf_scenario_progresses_toward_real_overstrain_rule() -> None:
    events = generate_osf_scenario(cycles=12)
    engineered = calculate_operating_features(events_to_operating_frame(events))

    assert _all_synthetic(events)
    assert events[0].uid is None
    assert events[0].product_id is None
    assert events[-1].tool_wear_min > events[0].tool_wear_min
    assert events[-1].torque_nm > events[0].torque_nm
    assert engineered.loc[0, OVERSTRAIN_LOAD] < engineered.loc[0, OVERSTRAIN_THRESHOLD]
    assert engineered.loc[len(engineered) - 1, OVERSTRAIN_LOAD] > engineered.loc[
        len(engineered) - 1, OVERSTRAIN_THRESHOLD
    ]
    assert bool(engineered.loc[0, OSF_RULE_CONDITION]) is False
    assert bool(engineered.loc[len(engineered) - 1, OSF_RULE_CONDITION]) is True


def test_hdf_scenario_progresses_toward_real_heat_dissipation_rule() -> None:
    events = generate_hdf_scenario(cycles=12)
    engineered = calculate_operating_features(events_to_operating_frame(events))

    assert _all_synthetic(events)
    assert engineered.loc[len(engineered) - 1, TEMPERATURE_DELTA] < engineered.loc[
        0,
        TEMPERATURE_DELTA,
    ]
    assert events[-1].rotational_speed_rpm < events[0].rotational_speed_rpm
    assert bool(engineered.loc[0, HDF_RULE_CONDITION]) is False
    assert bool(engineered.loc[len(engineered) - 1, HDF_RULE_CONDITION]) is True


def test_pwf_scenario_progresses_toward_real_power_rule() -> None:
    events = generate_pwf_scenario(cycles=12)
    engineered = calculate_operating_features(events_to_operating_frame(events))

    assert _all_synthetic(events)
    assert engineered.loc[0, MECHANICAL_POWER] < 9000
    assert engineered.loc[len(engineered) - 1, MECHANICAL_POWER] > 9000
    assert bool(engineered.loc[0, PWF_RULE_CONDITION]) is False
    assert bool(engineered.loc[len(engineered) - 1, PWF_RULE_CONDITION]) is True


def test_named_scenario_router_and_invalid_inputs() -> None:
    assert generate_scenario_events("osf", cycles=3)[0].source_label == SYNTHETIC_SCENARIO_LABEL
    assert generate_scenario_events("hdf", cycles=3)[0].source_label == SYNTHETIC_SCENARIO_LABEL
    assert generate_scenario_events("pwf", cycles=3)[0].source_label == SYNTHETIC_SCENARIO_LABEL

    with pytest.raises(ValueError, match="cycles"):
        generate_osf_scenario(cycles=1)
    with pytest.raises(ValueError, match="Unsupported scenario"):
        generate_scenario_events("bad")  # type: ignore[arg-type]


def _all_synthetic(events) -> bool:
    return all(
        event.source == "synthetic_demo_scenario"
        and event.source_label == SYNTHETIC_SCENARIO_LABEL
        for event in events
    )
