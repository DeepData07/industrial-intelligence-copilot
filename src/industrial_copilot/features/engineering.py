"""Deterministic engineering features derived from raw AI4I measurements."""

from __future__ import annotations

import math

import pandas as pd

from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    FAILURE_MODES,
    MACHINE_FAILURE,
    PROCESS_TEMPERATURE,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
    DataContractError,
    missing_expected_columns,
)

TEMPERATURE_DELTA = "Temperature delta [K]"
ANGULAR_VELOCITY = "Angular velocity [rad/s]"
MECHANICAL_POWER = "Mechanical power [W]"
OVERSTRAIN_LOAD = "Overstrain load [min Nm]"
OVERSTRAIN_THRESHOLD = "Overstrain threshold [min Nm]"
HDF_RULE_CONDITION = "HDF documented rule condition"
PWF_RULE_CONDITION = "PWF documented rule condition"
OSF_RULE_CONDITION = "OSF documented rule condition"
HDF_LABEL_AGREEMENT = "HDF rule-label agreement"
PWF_LABEL_AGREEMENT = "PWF rule-label agreement"
OSF_LABEL_AGREEMENT = "OSF rule-label agreement"
FAILURE_MODE_OR = "Failure mode OR"
MACHINE_FAILURE_AGREEMENT = "Machine failure-mode agreement"

ENGINEERING_COLUMNS = (
    TEMPERATURE_DELTA,
    ANGULAR_VELOCITY,
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
    OVERSTRAIN_THRESHOLD,
    HDF_RULE_CONDITION,
    PWF_RULE_CONDITION,
    OSF_RULE_CONDITION,
    HDF_LABEL_AGREEMENT,
    PWF_LABEL_AGREEMENT,
    OSF_LABEL_AGREEMENT,
    FAILURE_MODE_OR,
    MACHINE_FAILURE_AGREEMENT,
)

OPERATING_INPUT_COLUMNS = (
    PRODUCT_TYPE,
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    ROTATIONAL_SPEED,
    TORQUE,
    TOOL_WEAR,
)


def calculate_engineering_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copied frame with documented AI4I engineering features and rule checks.

    The source frame is not modified. TWF and RNF intentionally receive no deterministic
    condition because their documented mechanisms contain random components.
    """

    missing_columns = missing_expected_columns(frame.columns)
    if missing_columns:
        raise DataContractError(f"Cannot calculate engineering features; missing: {missing_columns}")
    engineered = calculate_operating_features(frame)

    engineered[HDF_LABEL_AGREEMENT] = engineered[HDF_RULE_CONDITION].eq(engineered["HDF"].eq(1))
    engineered[PWF_LABEL_AGREEMENT] = engineered[PWF_RULE_CONDITION].eq(engineered["PWF"].eq(1))
    engineered[OSF_LABEL_AGREEMENT] = engineered[OSF_RULE_CONDITION].eq(engineered["OSF"].eq(1))
    engineered[FAILURE_MODE_OR] = engineered[list(FAILURE_MODES)].eq(1).any(axis=1).astype(int)
    engineered[MACHINE_FAILURE_AGREEMENT] = engineered[MACHINE_FAILURE].eq(
        engineered[FAILURE_MODE_OR]
    )
    return engineered


def calculate_operating_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return operating/engineering features from telemetry without requiring outcome labels.

    This is used for leakage-safe prediction inputs. It intentionally does not create
    label-agreement columns because those depend on target/failure-mode labels unavailable
    at prediction time.
    """

    missing_columns = [column for column in OPERATING_INPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DataContractError(f"Cannot calculate operating features; missing: {missing_columns}")
    engineered = frame.copy(deep=True)
    engineered[TEMPERATURE_DELTA] = (
        engineered[PROCESS_TEMPERATURE] - engineered[AIR_TEMPERATURE]
    )
    engineered[ANGULAR_VELOCITY] = engineered[ROTATIONAL_SPEED] * (2 * math.pi / 60)
    engineered[MECHANICAL_POWER] = engineered[TORQUE] * engineered[ANGULAR_VELOCITY]
    engineered[OVERSTRAIN_LOAD] = engineered[TOOL_WEAR] * engineered[TORQUE]
    engineered[OVERSTRAIN_THRESHOLD] = engineered[PRODUCT_TYPE].map(
        {"L": 11000, "M": 12000, "H": 13000}
    )
    engineered[HDF_RULE_CONDITION] = (engineered[TEMPERATURE_DELTA] < 8.6) & (
        engineered[ROTATIONAL_SPEED] < 1380
    )
    engineered[PWF_RULE_CONDITION] = (engineered[MECHANICAL_POWER] < 3500) | (
        engineered[MECHANICAL_POWER] > 9000
    )
    engineered[OSF_RULE_CONDITION] = engineered[OVERSTRAIN_LOAD] > engineered[OVERSTRAIN_THRESHOLD]
    return engineered
