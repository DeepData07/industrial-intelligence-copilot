"""The explicit, immutable data contract for the AI4I source dataset."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

UID = "UID"
SOURCE_UID_ALIAS = "UDI"
PRODUCT_ID = "Product ID"
PRODUCT_TYPE = "Type"
AIR_TEMPERATURE = "Air temperature [K]"
PROCESS_TEMPERATURE = "Process temperature [K]"
ROTATIONAL_SPEED = "Rotational speed [rpm]"
TORQUE = "Torque [Nm]"
TOOL_WEAR = "Tool wear [min]"
MACHINE_FAILURE = "Machine failure"
FAILURE_MODES = ("TWF", "HDF", "PWF", "OSF", "RNF")

EXPECTED_COLUMNS = (
    UID,
    PRODUCT_ID,
    PRODUCT_TYPE,
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    ROTATIONAL_SPEED,
    TORQUE,
    TOOL_WEAR,
    MACHINE_FAILURE,
    *FAILURE_MODES,
)

NUMERIC_COLUMNS = (
    UID,
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    ROTATIONAL_SPEED,
    TORQUE,
    TOOL_WEAR,
    MACHINE_FAILURE,
    *FAILURE_MODES,
)

LABEL_COLUMNS = (MACHINE_FAILURE, *FAILURE_MODES)
VALID_PRODUCT_TYPES = frozenset({"L", "M", "H"})


class DataContractError(ValueError):
    """Raised when source data cannot be safely used for analysis."""


def missing_expected_columns(columns: Iterable[str]) -> list[str]:
    """Return required column names absent from a supplied header."""

    actual = set(columns)
    return [column for column in EXPECTED_COLUMNS if column not in actual]


def unexpected_columns(columns: Iterable[str]) -> list[str]:
    """Return supplied column names that are outside the source contract."""

    expected = set(EXPECTED_COLUMNS)
    return [column for column in columns if column not in expected]


def assert_expected_columns(frame: pd.DataFrame) -> None:
    """Reject files whose schema does not exactly match the published AI4I contract."""

    missing = missing_expected_columns(frame.columns)
    unexpected = unexpected_columns(frame.columns)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise DataContractError("AI4I schema mismatch: " + "; ".join(details))
