from __future__ import annotations

import math

import pandas as pd
import pytest

from industrial_copilot.features.engineering import (
    HDF_LABEL_AGREEMENT,
    HDF_RULE_CONDITION,
    MACHINE_FAILURE_AGREEMENT,
    MECHANICAL_POWER,
    OSF_LABEL_AGREEMENT,
    OSF_RULE_CONDITION,
    OVERSTRAIN_LOAD,
    OVERSTRAIN_THRESHOLD,
    PWF_LABEL_AGREEMENT,
    PWF_RULE_CONDITION,
    TEMPERATURE_DELTA,
    calculate_engineering_features,
)


def test_engineering_formulas_and_documented_rules(sample_ai4i_frame) -> None:
    engineered = calculate_engineering_features(sample_ai4i_frame)

    assert engineered.loc[0, TEMPERATURE_DELTA] == 10.0
    assert engineered.loc[0, MECHANICAL_POWER] == pytest.approx(40.0 * 1500 * 2 * math.pi / 60)
    assert engineered.loc[3, OVERSTRAIN_LOAD] == 11500.0
    assert engineered.loc[3, OVERSTRAIN_THRESHOLD] == 11000
    assert bool(engineered.loc[1, HDF_RULE_CONDITION]) is True
    assert bool(engineered.loc[2, PWF_RULE_CONDITION]) is True
    assert bool(engineered.loc[3, OSF_RULE_CONDITION]) is True
    assert engineered[HDF_LABEL_AGREEMENT].all()
    assert engineered[PWF_LABEL_AGREEMENT].all()
    assert engineered[OSF_LABEL_AGREEMENT].all()
    assert engineered[MACHINE_FAILURE_AGREEMENT].all()


def test_feature_calculation_does_not_mutate_raw_frame(sample_ai4i_frame) -> None:
    before = sample_ai4i_frame.copy(deep=True)

    calculate_engineering_features(sample_ai4i_frame)

    pd.testing.assert_frame_equal(sample_ai4i_frame, before)
    assert TEMPERATURE_DELTA not in sample_ai4i_frame.columns
