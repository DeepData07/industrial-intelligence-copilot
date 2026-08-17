from __future__ import annotations

import pandas as pd

from industrial_copilot.data.audit import run_data_contract_audit
from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    EXPECTED_COLUMNS,
    MACHINE_FAILURE,
    PRODUCT_TYPE,
    TORQUE,
    UID,
)


def valid_frame() -> pd.DataFrame:
    """Small source-shaped data that follows documented deterministic rules."""

    return pd.DataFrame(
        [
            [1, "L00001", "L", 300.0, 310.0, 1500, 40.0, 10, 0, 0, 0, 0, 0, 0],
            [2, "M00002", "M", 300.0, 308.0, 1300, 40.0, 10, 1, 0, 1, 0, 0, 0],
            [3, "H00003", "H", 300.0, 310.0, 2000, 10.0, 10, 1, 0, 0, 1, 0, 0],
            [4, "L00004", "L", 300.0, 310.0, 1500, 50.0, 230, 1, 0, 0, 0, 1, 0],
        ],
        columns=EXPECTED_COLUMNS,
    )


def test_valid_contract_passes_and_audits_documented_rules() -> None:
    audit = run_data_contract_audit(valid_frame())

    assert audit.passed is True
    assert audit.row_count == 4
    assert audit.duplicate_uid_count == 0
    assert audit.failure_rule_consistency["HDF"].false_positive_labels == 0
    assert audit.failure_rule_consistency["PWF"].false_negative_labels == 0
    assert audit.failure_rule_consistency["OSF"].matching_positive_rows == 1
    assert audit.failure_rule_consistency["TWF"].validation_status == "not_applicable"
    assert audit.machine_failure_or_mismatch_count == 0
    assert audit.machine_failure_positive_without_mode_count == 0
    assert audit.machine_failure_negative_with_mode_count == 0


def test_audit_reports_violations_without_changing_the_input() -> None:
    frame = valid_frame()
    frame.loc[1, PRODUCT_TYPE] = "X"
    frame.loc[2, UID] = 1
    frame.loc[3, TORQUE] = -1.0
    frame.loc[0, MACHINE_FAILURE] = 1
    before = frame.copy(deep=True)

    audit = run_data_contract_audit(frame)

    checks = {issue.check for issue in audit.issues}
    assert audit.passed is False
    assert {"product_type", "uid_uniqueness", f"numeric_sanity:{TORQUE}", "machine_failure_or"} <= checks
    assert audit.machine_failure_positive_without_mode_count == 1
    pd.testing.assert_frame_equal(frame, before)


def test_audit_reports_missing_values_and_label_shape_errors() -> None:
    frame = valid_frame()
    frame.loc[0, AIR_TEMPERATURE] = None
    frame.loc[1, "HDF"] = 2

    audit = run_data_contract_audit(frame)

    checks = {issue.check for issue in audit.issues}
    assert "missing_values" in checks
    assert "binary_label:HDF" in checks
    assert audit.missing_values[AIR_TEMPERATURE] == 1
