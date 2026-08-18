"""Incident-aware deterministic copilot responses for live operations."""

from __future__ import annotations

import re

from industrial_copilot.analytics.models import AnalysisFilters
from industrial_copilot.copilot.schemas import (
    EvidenceFinding,
    EvidenceMetric,
    EvidencePackage,
)
from industrial_copilot.data.schema import ROTATIONAL_SPEED, TOOL_WEAR, TORQUE
from industrial_copilot.features.engineering import MECHANICAL_POWER, OVERSTRAIN_LOAD
from industrial_copilot.simulation.investigation import (
    FeatureChange,
    IncidentInvestigationPackage,
)


def answer_incident_question(
    question: str,
    package: IncidentInvestigationPackage,
) -> tuple[str, EvidencePackage]:
    """Answer an incident follow-up from compact deterministic incident evidence."""

    normalized = question.strip().lower()
    findings = _findings_for(normalized, package)
    metrics = _metrics_for(package)
    evidence = EvidencePackage(
        question=question,
        intent="incident_investigation",
        filters=AnalysisFilters(),
        sample_size=package.what_changed.recent_observation_count,
        calculations_run=[
            "incident_context_package",
            "calculate_what_changed",
            "find_similar_historical_conditions_for_event",
        ],
        findings=findings,
        metrics=metrics,
        visualisations=[
            "What Changed comparison",
            "Similar historical conditions summary",
        ],
        statistical_tests=[],
        model_evidence=[],
        engineering_evidence=[package.model_dump()],
        data_quality_warnings=[],
        uncertainty=[
            "Incident evidence is decision support; it does not prove causality.",
            "The live stream is simulated and should not be interpreted as real PLC history.",
        ],
        limitations=list(package.limitations),
        suggested_next_questions=[
            "Has this happened under similar conditions before?",
            "Is RPM contributing?",
            "What should I inspect first?",
        ],
        tool_results={"incident_investigation_package": package.model_dump()},
    )
    return _render_incident_answer(findings), evidence


def _findings_for(
    normalized_question: str,
    package: IncidentInvestigationPackage,
) -> list[EvidenceFinding]:
    if _asks_similarity(normalized_question):
        return [_similarity_finding(package)]
    if _asks_rpm(normalized_question):
        return [_rpm_finding(package)]
    if _asks_adjustment(normalized_question):
        return [_adjustment_finding(package)]
    if _asks_action(normalized_question):
        return [_action_finding(package)]
    if _asks_confidence(normalized_question):
        return [_confidence_finding(package)]
    return [_flag_reason_finding(package), _largest_change_finding(package), _similarity_finding(package)]


def _flag_reason_finding(package: IncidentInvestigationPackage) -> EvidenceFinding:
    return EvidenceFinding(
        statement=(
            f"Incident {package.incident_id} is active for {package.asset_id}; "
            f"{package.what_changed.summary}"
        ),
        source_tools=["incident_context_package", "calculate_what_changed"],
    )


def _largest_change_finding(package: IncidentInvestigationPackage) -> EvidenceFinding:
    top = package.what_changed.largest_changes[:3]
    if not top:
        statement = package.what_changed.summary
    else:
        statement = "Largest recent changes: " + "; ".join(_format_change(item) for item in top) + "."
    return EvidenceFinding(statement=statement, source_tools=["calculate_what_changed"])


def _similarity_finding(package: IncidentInvestigationPackage) -> EvidenceFinding:
    similar = package.similar_historical_conditions
    if similar.returned_observation_count == 0 or similar.similar_case_failure_rate is None:
        statement = "No similar historical AI4I observations were available for this live condition."
    else:
        mode = (
            f" Most common associated failure flag: {similar.most_common_failure_mode}."
            if similar.most_common_failure_mode
            else ""
        )
        statement = (
            f"Similar historical conditions: {similar.returned_observation_count} retrieved; "
            f"{similar.failed_observation_count} failed "
            f"({similar.similar_case_failure_rate:.1%}).{mode}"
        )
    return EvidenceFinding(statement=statement, source_tools=["find_similar_historical_conditions_for_event"])


def _rpm_finding(package: IncidentInvestigationPackage) -> EvidenceFinding:
    rpm = _change_by_feature(package, ROTATIONAL_SPEED)
    largest = package.what_changed.largest_changes[0] if package.what_changed.largest_changes else None
    if rpm is None:
        statement = "RPM contribution cannot be compared yet because the live baseline window is incomplete."
    elif largest is not None and largest.feature != ROTATIONAL_SPEED:
        statement = (
            f"{ROTATIONAL_SPEED} (RPM) {rpm.direction} by {_change_amount(rpm)}, but the strongest recent change is "
            f"{largest.feature} ({_change_amount(largest)}). Current evidence does not support RPM "
            "as the primary driver of this alert."
        )
    else:
        statement = (
            f"{ROTATIONAL_SPEED} (RPM) is one of the strongest observed changes in the current window "
            f"({_change_amount(rpm)}). This is association evidence, not proof of causation."
        )
    return EvidenceFinding(statement=statement, source_tools=["calculate_what_changed"])


def _action_finding(package: IncidentInvestigationPackage) -> EvidenceFinding:
    top_features = {item.feature for item in package.what_changed.largest_changes[:3]}
    checks = []
    if TORQUE in top_features or OVERSTRAIN_LOAD in top_features or TOOL_WEAR in top_features:
        checks.append("inspect current tool condition")
        checks.append("verify whether the torque increase is expected for this production cycle")
    if MECHANICAL_POWER in top_features:
        checks.append("review the RPM-torque operating point and expected power demand")
    if not checks:
        checks.append("compare current telemetry with the recent healthy baseline")
    statement = "Suggested next checks: " + "; ".join(checks) + "."
    return EvidenceFinding(statement=statement, source_tools=["incident_context_package"])


def _adjustment_finding(package: IncidentInvestigationPackage) -> EvidenceFinding:
    if not package.adjustment_options:
        statement = (
            "No exact parameter adjustment is supported by the current evidence. Use the "
            "backend What-if analysis and obtain engineer approval before changing operation."
        )
    else:
        options = []
        for option in package.adjustment_options:
            if option.parameter == "Torque":
                options.append(
                    f"reduce torque from {option.current_value:.1f} to no more than "
                    f"{option.proposed_value:.1f} {option.unit} "
                    f"(a reduction of {option.change_amount:.1f} {option.unit}, "
                    f"{option.change_percent:.1%})"
                )
            else:
                options.append(
                    f"{option.action} so {option.parameter.casefold()} moves from "
                    f"{option.current_value:.1f} to no more than {option.proposed_value:.1f} "
                    f"{option.unit}"
                )
        expected_margin = min(
            option.expected_osf_margin_min_nm for option in package.adjustment_options
        )
        statement = (
            "Rule-based OSF decision-support options: "
            + "; or ".join(options)
            + f". Each option targets an OSF margin of at least {expected_margin:.0f} min Nm. "
            "Validate the proposal in What-if analysis and obtain engineer approval; this is "
            "not a machine command or a guarantee that model risk will clear."
        )
    return EvidenceFinding(
        statement=statement,
        source_tools=["calculate_rule_based_adjustment", "incident_context_package"],
    )


def _confidence_finding(package: IncidentInvestigationPackage) -> EvidenceFinding:
    similar = package.similar_historical_conditions
    if similar.similar_case_failure_rate is None:
        statement = "Confidence is limited because similar historical evidence is unavailable."
    else:
        statement = (
            f"Confidence should remain cautious: similar AI4I conditions failed at "
            f"{similar.similar_case_failure_rate:.1%}, and the scenario/live layer is simulated."
        )
    return EvidenceFinding(statement=statement, source_tools=["find_similar_historical_conditions_for_event"])


def _metrics_for(package: IncidentInvestigationPackage) -> list[EvidenceMetric]:
    metrics = [
        EvidenceMetric(
            label="Recent window",
            value=str(package.what_changed.recent_observation_count),
            source_tool="calculate_what_changed",
        ),
        EvidenceMetric(
            label="Baseline window",
            value=str(package.what_changed.baseline_observation_count),
            source_tool="calculate_what_changed",
        ),
        EvidenceMetric(
            label="Similar observations",
            value=str(package.similar_historical_conditions.returned_observation_count),
            source_tool="find_similar_historical_conditions_for_event",
        ),
    ]
    rate = package.similar_historical_conditions.similar_case_failure_rate
    if rate is not None:
        metrics.append(
            EvidenceMetric(
                label="Similar-case failure rate",
                value=f"{rate:.1%}",
                source_tool="find_similar_historical_conditions_for_event",
            )
        )
    return metrics


def _render_incident_answer(findings: list[EvidenceFinding]) -> str:
    return " ".join(finding.statement for finding in findings[:3])


def _format_change(change: FeatureChange) -> str:
    return f"{change.feature} {change.direction} by {_change_amount(change)}"


def _change_amount(change: FeatureChange) -> str:
    if change.percent_change is not None:
        return f"{change.percent_change:.1%}"
    return f"{change.absolute_change:.2f}"


def _change_by_feature(
    package: IncidentInvestigationPackage,
    feature: str,
) -> FeatureChange | None:
    for change in package.what_changed.changes:
        if change.feature == feature:
            return change
    return None


def _asks_similarity(question: str) -> bool:
    return bool(re.search(r"\bsimilar|happened before|before|historical|history\b", question))


def _asks_rpm(question: str) -> bool:
    return bool(re.search(r"\brpm|rotational speed|speed\b", question))


def _asks_action(question: str) -> bool:
    return bool(re.search(r"\binspect|check|next|action|do first\b", question))


def _asks_adjustment(question: str) -> bool:
    return bool(
        re.search(
            r"\b(by how much|what (?:value|parameter)|adjust|reduce|lower|increase|"
            r"change.*(?:resolve|clear|fix)|parameter.*change)\b",
            question,
        )
    )


def _asks_confidence(question: str) -> bool:
    return bool(re.search(r"\bconfident|confidence|sure|trust\b", question))
