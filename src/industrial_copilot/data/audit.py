"""Deterministic data-contract audit for AI4I; findings never alter source data."""

from __future__ import annotations

from collections import Counter
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    FAILURE_MODES,
    LABEL_COLUMNS,
    MACHINE_FAILURE,
    NUMERIC_COLUMNS,
    PROCESS_TEMPERATURE,
    PRODUCT_ID,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
    UID,
    VALID_PRODUCT_TYPES,
    assert_expected_columns,
)

IssueSeverity = Literal["error", "warning", "info"]


class AuditIssue(BaseModel):
    """A transparent, count-based data-contract finding."""

    model_config = ConfigDict(frozen=True)

    check: str
    severity: IssueSeverity
    message: str
    affected_rows: int | None = Field(default=None, ge=0)


class FailureRuleAudit(BaseModel):
    """Agreement between a documented deterministic rule and its published label."""

    model_config = ConfigDict(frozen=True)

    failure_mode: str
    validation_status: Literal["checked", "not_applicable"]
    documented_rule_matches: int | None = Field(default=None, ge=0)
    label_positive_rows: int = Field(ge=0)
    matching_positive_rows: int | None = Field(default=None, ge=0)
    false_positive_labels: int | None = Field(default=None, ge=0)
    false_negative_labels: int | None = Field(default=None, ge=0)
    note: str


class DataContractAudit(BaseModel):
    """Structured audit result consumed later by APIs and the evidence package."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    missing_values: dict[str, int]
    duplicate_row_count: int = Field(ge=0)
    duplicate_uid_count: int = Field(ge=0)
    product_type_counts: dict[str, int]
    numeric_ranges: dict[str, dict[str, float | int | None]]
    label_distribution: dict[str, dict[str, float | int]]
    failure_rule_consistency: dict[str, FailureRuleAudit]
    machine_failure_or_mismatch_count: int = Field(ge=0)
    machine_failure_positive_without_mode_count: int = Field(ge=0)
    machine_failure_negative_with_mode_count: int = Field(ge=0)
    issues: list[AuditIssue]


def _binary_label_issues(frame: pd.DataFrame) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for column in LABEL_COLUMNS:
        invalid_count = int((~frame[column].isin([0, 1])).sum())
        if invalid_count:
            issues.append(
                AuditIssue(
                    check=f"binary_label:{column}",
                    severity="error",
                    message=f"{column} must contain only 0 or 1.",
                    affected_rows=invalid_count,
                )
            )
    return issues


def _numeric_sanity_issues(frame: pd.DataFrame) -> list[AuditIssue]:
    """Check physical validity, not arbitrary upper bounds that would hide extreme operation."""

    rules = {
        UID: (frame[UID] <= 0, "UID must be positive."),
        AIR_TEMPERATURE: (frame[AIR_TEMPERATURE] <= 0, "Air temperature must be above 0 K."),
        PROCESS_TEMPERATURE: (
            frame[PROCESS_TEMPERATURE] <= 0,
            "Process temperature must be above 0 K.",
        ),
        ROTATIONAL_SPEED: (frame[ROTATIONAL_SPEED] <= 0, "Rotational speed must be positive."),
        TORQUE: (frame[TORQUE] < 0, "Torque cannot be negative."),
        TOOL_WEAR: (frame[TOOL_WEAR] < 0, "Tool wear cannot be negative."),
    }
    issues: list[AuditIssue] = []
    for column, (invalid_rows, message) in rules.items():
        count = int(invalid_rows.sum())
        if count:
            issues.append(
                AuditIssue(
                    check=f"numeric_sanity:{column}",
                    severity="error",
                    message=message,
                    affected_rows=count,
                )
            )
    return issues


def _failure_rule_audit(frame: pd.DataFrame) -> dict[str, FailureRuleAudit]:
    temperature_delta = frame[PROCESS_TEMPERATURE] - frame[AIR_TEMPERATURE]
    mechanical_power = frame[TORQUE] * frame[ROTATIONAL_SPEED] * (2 * 3.141592653589793 / 60)
    overstrain_threshold = frame[PRODUCT_TYPE].map({"L": 11000, "M": 12000, "H": 13000})

    documented_rules = {
        "HDF": (temperature_delta < 8.6) & (frame[ROTATIONAL_SPEED] < 1380),
        "PWF": (mechanical_power < 3500) | (mechanical_power > 9000),
        "OSF": (frame[TOOL_WEAR] * frame[TORQUE]) > overstrain_threshold,
    }
    result: dict[str, FailureRuleAudit] = {
        "TWF": FailureRuleAudit(
            failure_mode="TWF",
            validation_status="not_applicable",
            label_positive_rows=int(frame["TWF"].sum()),
            note=(
                "TWF uses a randomly selected replacement/failure point; a deterministic rule "
                "comparison would be misleading."
            ),
        ),
        "RNF": FailureRuleAudit(
            failure_mode="RNF",
            validation_status="not_applicable",
            label_positive_rows=int(frame["RNF"].sum()),
            note="RNF is a random mechanism and has no documented deterministic predicate.",
        ),
    }
    for mode, rule in documented_rules.items():
        labels = frame[mode].eq(1)
        result[mode] = FailureRuleAudit(
            failure_mode=mode,
            validation_status="checked",
            documented_rule_matches=int(rule.sum()),
            label_positive_rows=int(labels.sum()),
            matching_positive_rows=int((rule & labels).sum()),
            false_positive_labels=int((labels & ~rule).sum()),
            false_negative_labels=int((rule & ~labels).sum()),
            note="Compared against the documented AI4I generation rule; no labels were changed.",
        )
    return result


def _numeric_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    ranges: dict[str, dict[str, float | int | None]] = {}
    for column in NUMERIC_COLUMNS:
        values = frame[column]
        if not pd.api.types.is_numeric_dtype(values) or values.dropna().empty:
            ranges[column] = {"min": None, "max": None}
            continue
        ranges[column] = {
            "min": _python_number(values.min()),
            "max": _python_number(values.max()),
        }
    return ranges


def _python_number(value: object) -> float | int:
    """Convert NumPy scalar values into JSON-compatible standard numeric values."""

    return value.item() if hasattr(value, "item") else value  # type: ignore[return-value]


def run_data_contract_audit(frame: pd.DataFrame) -> DataContractAudit:
    """Audit AI4I structure, values, labels, and documented-rule consistency."""

    assert_expected_columns(frame)
    issues: list[AuditIssue] = []
    missing_values = {column: int(count) for column, count in frame.isna().sum().items()}
    missing_total = sum(missing_values.values())
    if missing_total:
        issues.append(
            AuditIssue(
                check="missing_values",
                severity="error",
                message="AI4I source data contains missing values; no imputation was applied.",
                affected_rows=missing_total,
            )
        )

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        issues.append(
            AuditIssue(
                check="duplicate_rows",
                severity="warning",
                message="Exact duplicate observations were found; raw rows were preserved.",
                affected_rows=duplicate_rows,
            )
        )

    duplicate_uids = int(frame[UID].duplicated(keep=False).sum())
    if duplicate_uids:
        issues.append(
            AuditIssue(
                check="uid_uniqueness",
                severity="error",
                message="UID must uniquely identify an observation.",
                affected_rows=duplicate_uids,
            )
        )

    invalid_types = int((~frame[PRODUCT_TYPE].isin(VALID_PRODUCT_TYPES)).sum())
    if invalid_types:
        issues.append(
            AuditIssue(
                check="product_type",
                severity="error",
                message="Type must be one of L, M, or H.",
                affected_rows=invalid_types,
            )
        )

    product_id_prefix_mismatch = int(
        (~frame[PRODUCT_ID].astype(str).str.startswith(tuple(VALID_PRODUCT_TYPES))).sum()
    )
    if product_id_prefix_mismatch:
        issues.append(
            AuditIssue(
                check="product_id_prefix",
                severity="warning",
                message="Product ID does not start with a recognized L/M/H quality prefix.",
                affected_rows=product_id_prefix_mismatch,
            )
        )

    product_id_type_mismatch = int(
        (frame[PRODUCT_ID].astype(str).str[0] != frame[PRODUCT_TYPE].astype(str)).sum()
    )
    if product_id_type_mismatch:
        issues.append(
            AuditIssue(
                check="product_id_type_consistency",
                severity="warning",
                message="Product ID quality prefix differs from the Type category.",
                affected_rows=product_id_type_mismatch,
            )
        )

    non_numeric_columns = [
        column for column in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    for column in non_numeric_columns:
        issues.append(
            AuditIssue(
                check=f"numeric_dtype:{column}",
                severity="error",
                message=f"{column} must be numeric.",
                affected_rows=None,
            )
        )
    if not non_numeric_columns:
        issues.extend(_binary_label_issues(frame))
        issues.extend(_numeric_sanity_issues(frame))

    failure_rules = _failure_rule_audit(frame) if not non_numeric_columns else {}
    for mode, rule_audit in failure_rules.items():
        if rule_audit.validation_status == "checked":
            mismatches = rule_audit.false_positive_labels + rule_audit.false_negative_labels  # type: ignore[operator]
            if mismatches:
                issues.append(
                    AuditIssue(
                        check=f"failure_rule:{mode}",
                        severity="warning",
                        message=f"{mode} labels differ from the documented deterministic rule.",
                        affected_rows=mismatches,
                    )
                )

    failure_mode_or = frame[list(FAILURE_MODES)].eq(1).any(axis=1).astype(int)
    machine_failure_positive_without_mode = int(
        ((frame[MACHINE_FAILURE] == 1) & (failure_mode_or == 0)).sum()
    )
    machine_failure_negative_with_mode = int(
        ((frame[MACHINE_FAILURE] == 0) & (failure_mode_or == 1)).sum()
    )
    machine_failure_mismatches = int((frame[MACHINE_FAILURE] != failure_mode_or).sum())
    if machine_failure_mismatches:
        issues.append(
            AuditIssue(
                check="machine_failure_or",
                severity="warning",
                message=(
                    "Machine failure differs from the OR of failure-mode labels; source labels "
                    "were preserved for investigation."
                ),
                affected_rows=machine_failure_mismatches,
            )
        )

    label_distribution = {
        column: {
            "positive_count": int(frame[column].eq(1).sum()),
            "positive_rate": float(frame[column].eq(1).mean()) if len(frame) else 0.0,
        }
        for column in LABEL_COLUMNS
    }
    product_counts = dict(sorted(Counter(frame[PRODUCT_TYPE]).items()))

    return DataContractAudit(
        passed=not any(issue.severity == "error" for issue in issues),
        row_count=len(frame),
        column_count=len(frame.columns),
        missing_values=missing_values,
        duplicate_row_count=duplicate_rows,
        duplicate_uid_count=duplicate_uids,
        product_type_counts=product_counts,
        numeric_ranges=_numeric_ranges(frame),
        label_distribution=label_distribution,
        failure_rule_consistency=failure_rules,
        machine_failure_or_mismatch_count=machine_failure_mismatches,
        machine_failure_positive_without_mode_count=machine_failure_positive_without_mode,
        machine_failure_negative_with_mode_count=machine_failure_negative_with_mode,
        issues=issues,
    )
