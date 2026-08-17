"""Deterministic keyword-and-context planner for offline evidence routing."""

from __future__ import annotations

import re

from industrial_copilot.analytics.models import AnalysisFilters, NumericRange
from industrial_copilot.copilot.schemas import ToolCall, ToolPlan
from industrial_copilot.copilot.state import ConversationState
from industrial_copilot.data.schema import ROTATIONAL_SPEED, TOOL_WEAR, TORQUE

UNAVAILABLE_PATTERNS = {
    "timestamped history": r"\b(last|past)\s+\d+\s+(day|days|week|weeks|month|months)\b|\bhistory\b",
    "vibration": r"\bvibration\b",
    "pressure": r"\bpressure\b",
    "downtime": r"\bdowntime\b|\bmttr\b|\brepair duration\b",
    "energy": r"\benergy\b|\belectric(?:al)? usage\b|\bpower consumption\b",
    "maintenance actions": r"\bmaintenance action\b|\brepair action\b",
}

VARIABLE_KEYWORDS = {
    "torque": TORQUE,
    "tool wear": TOOL_WEAR,
    "rpm": ROTATIONAL_SPEED,
    "rotational speed": ROTATIONAL_SPEED,
}


def plan_offline_question(question: str, state: ConversationState) -> ToolPlan:
    """Translate supported AI4I questions into a fixed plan with no free-form execution path."""

    normalized = question.strip().lower()
    if not normalized:
        return ToolPlan(intent="unsupported_question", clarification="Please ask a question about AI4I data.")
    unavailable = _unavailable_topic(normalized)
    if unavailable is not None:
        return ToolPlan(
            intent="unavailable_data",
            clarification=(
                f"AI4I does not contain {unavailable}. It has cross-sectional synthetic observations, "
                "not genuine timestamped machine histories or additional plant sensors."
            ),
        )

    filters = _merge_filters(_extract_filters(normalized), state.current_filters)
    uid = _extract_uid(normalized)
    variable = _extract_variable(normalized) or state.current_variable
    filter_only = _is_filter_only(normalized)

    if "data trust" in normalized or "audit" in normalized or "data quality" in normalized:
        return ToolPlan(intent="data_quality", tools=[ToolCall(name="run_data_contract_audit")])
    if uid is not None and ("similar" in normalized or "like this" in normalized):
        return ToolPlan(
            intent="similar_conditions",
            tools=[ToolCall(name="find_similar_conditions", arguments={"uid": uid, "filters": filters.model_dump()})],
            state_updates=filters,
        )
    if uid is not None:
        return ToolPlan(
            intent="observation_lookup",
            tools=[
                ToolCall(name="get_observation", arguments={"uid": uid}),
                ToolCall(name="calculate_engineering_features", arguments={"uid": uid}),
            ],
            state_updates=filters,
        )
    if _is_failure_investigation(normalized):
        return ToolPlan(
            intent="failure_investigation",
            tools=[
                ToolCall(
                    name="failure_rate_by_range",
                    arguments={
                        "variable": ROTATIONAL_SPEED,
                        "ranges": [
                            {"column": ROTATIONAL_SPEED, "maximum": 1599},
                            {"column": ROTATIONAL_SPEED, "minimum": 1600},
                        ],
                        "filters": filters.model_dump(),
                    },
                ),
                ToolCall(
                    name="failure_mode_breakdown",
                    arguments={
                        "filters": filters.model_copy(
                            update={
                                "numeric_ranges": [
                                    *filters.numeric_ranges,
                                    NumericRange(column=ROTATIONAL_SPEED, minimum=1600),
                                ]
                            }
                        ).model_dump()
                    },
                ),
                ToolCall(
                    name="analyze_conditional_relationship",
                    arguments={
                        "exposure": ROTATIONAL_SPEED,
                        "exposure_threshold": 1600,
                        "conditioning_variable": TORQUE,
                        "filters": filters.model_dump(),
                    },
                ),
                ToolCall(name="discover_high_risk_regimes", arguments={"filters": filters.model_dump()}),
            ],
            state_updates=filters,
        )
    if filter_only and state.previous_intent is not None:
        return _follow_up_plan(state.previous_intent, filters, variable)
    if "product" in normalized or ("compare" in normalized and re.search(r"\b[lhm]\b", normalized)):
        return ToolPlan(
            intent="product_comparison",
            tools=[ToolCall(name="compare_product_types", arguments={"filters": filters.model_dump()})],
            state_updates=filters,
        )
    if "compare" in normalized and ("failed" in normalized or "healthy" in normalized or variable):
        return ToolPlan(
            intent="failed_healthy_comparison",
            tools=[
                ToolCall(
                    name="compare_failed_vs_healthy",
                    arguments={"variables": [variable or TORQUE], "filters": filters.model_dump()},
                )
            ],
            state_updates=filters,
        )
    if "failure rate" in normalized and ("rpm" in normalized or "speed" in normalized):
        return ToolPlan(
            intent="failure_rate",
            tools=[
                ToolCall(
                    name="failure_rate_by_range",
                    arguments={
                        "variable": ROTATIONAL_SPEED,
                        "ranges": [
                            {"column": ROTATIONAL_SPEED, "maximum": 1379},
                            {"column": ROTATIONAL_SPEED, "minimum": 1380, "maximum": 1600},
                            {"column": ROTATIONAL_SPEED, "minimum": 1601},
                        ],
                        "filters": filters.model_dump(),
                    },
                )
            ],
            state_updates=filters,
        )
    if "percentage" in normalized or "how many failed" in normalized or "failure rate" in normalized:
        return ToolPlan(
            intent="failure_rate",
            tools=[ToolCall(name="get_failure_rate", arguments={"filters": filters.model_dump()})],
            state_updates=filters,
        )
    if "what has" in normalized or "summary" in normalized or "happening" in normalized:
        return ToolPlan(
            intent="dataset_summary",
            tools=[ToolCall(name="get_dataset_summary", arguments={"filters": filters.model_dump()})],
            state_updates=filters,
        )
    return ToolPlan(
        intent="unsupported_question",
        clarification=(
            "I can analyze AI4I failure rates, failed-versus-healthy conditions, product types, "
            "RPM relationships, UID observations, similar conditions, or data quality."
        ),
    )


def _follow_up_plan(intent: str, filters: AnalysisFilters, variable: str | None) -> ToolPlan:
    if intent == "failed_healthy_comparison":
        return ToolPlan(
            intent="failed_healthy_comparison",
            tools=[
                ToolCall(
                    name="compare_failed_vs_healthy",
                    arguments={"variables": [variable or TORQUE], "filters": filters.model_dump()},
                )
            ],
            state_updates=filters,
        )
    if intent == "product_comparison":
        return ToolPlan(
            intent="product_comparison",
            tools=[ToolCall(name="compare_product_types", arguments={"filters": filters.model_dump()})],
            state_updates=filters,
        )
    return ToolPlan(
        intent="failure_rate",
        tools=[ToolCall(name="get_failure_rate", arguments={"filters": filters.model_dump()})],
        state_updates=filters,
    )


def _extract_filters(question: str) -> AnalysisFilters:
    product_match = re.search(r"\bonly\s+([lmh])(?:\s+products?)?\b", question)
    product_types = [product_match.group(1).upper()] if product_match else None
    machine_failure = 1 if re.search(r"\bonly\s+failed\b", question) else None
    if re.search(r"\bonly\s+healthy\b", question):
        machine_failure = 0
    ranges: list[NumericRange] = []
    rpm_match = re.search(r"\b(?:above|over|greater than|>=)\s+(\d+(?:\.\d+)?)\s*rpm\b", question)
    if rpm_match:
        ranges.append(NumericRange(column=ROTATIONAL_SPEED, minimum=float(rpm_match.group(1))))
    return AnalysisFilters(product_types=product_types, machine_failure=machine_failure, numeric_ranges=ranges)


def _merge_filters(new: AnalysisFilters, previous: AnalysisFilters) -> AnalysisFilters:
    """Keep prior context unless the follow-up explicitly changes a filter dimension."""

    new_ranges_by_column = {item.column: item for item in new.numeric_ranges}
    retained_ranges = [
        item for item in previous.numeric_ranges if item.column not in new_ranges_by_column
    ]
    return AnalysisFilters(
        product_types=new.product_types if new.product_types is not None else previous.product_types,
        machine_failure=(
            new.machine_failure if new.machine_failure is not None else previous.machine_failure
        ),
        failure_mode=new.failure_mode or previous.failure_mode,
        numeric_ranges=[*retained_ranges, *new.numeric_ranges],
    )


def _extract_variable(question: str) -> str | None:
    for keyword, variable in VARIABLE_KEYWORDS.items():
        if keyword in question:
            return variable
    return None


def _extract_uid(question: str) -> int | None:
    match = re.search(r"\b(?:uid|observation)\s*#?\s*(\d+)\b", question)
    return int(match.group(1)) if match else None


def _unavailable_topic(question: str) -> str | None:
    for topic, pattern in UNAVAILABLE_PATTERNS.items():
        if re.search(pattern, question):
            return topic
    return None


def _is_failure_investigation(question: str) -> bool:
    return bool(
        "why" in question
        and ("failure" in question or "failures" in question)
        and ("rpm" in question or "speed" in question)
    )


def _is_filter_only(question: str) -> bool:
    return bool(
        re.fullmatch(r"(?:only\s+)?(?:[lmh]\s*products?|failed|healthy|above\s+\d+(?:\.\d+)?\s*rpm)[.?!]*", question)
    )
