"""Evidence assembly for the deterministic offline industrial copilot."""

from __future__ import annotations

from typing import Any

from industrial_copilot.analytics.models import AnalysisFilters
from industrial_copilot.copilot.schemas import (
    EvidenceFinding,
    EvidenceMetric,
    EvidencePackage,
    ToolPlan,
)

LIMITATIONS = [
    "AI4I provides observation IDs, not timestamped machine histories.",
    "This is a synthetic cross-sectional benchmark dataset; observed associations are not causal claims.",
]


def build_evidence_package(
    question: str,
    plan: ToolPlan,
    results: dict[str, Any],
    filters: AnalysisFilters,
) -> EvidencePackage:
    """Create a traceable, display-ready package strictly from registered tool output."""

    findings: list[EvidenceFinding] = []
    metrics: list[EvidenceMetric] = []
    statistical_tests: list[dict[str, Any]] = []
    model_evidence: list[dict[str, Any]] = []
    engineering_evidence: list[dict[str, Any]] = []
    warnings: list[str] = []
    uncertainty: list[str] = []
    sample_size = _sample_size(results)

    failure = results.get("get_failure_rate")
    if failure:
        rate = failure["failure_rate"]
        findings.append(
            EvidenceFinding(
                statement=(
                    f"{failure['failed_observation_count']} of {failure['observation_count']} selected "
                    f"observations failed ({rate:.2%})."
                ),
                source_tools=["get_failure_rate"],
            )
        )
        metrics.extend(
            [
                EvidenceMetric(label="Selected observations", value=str(failure["observation_count"]), source_tool="get_failure_rate"),
                EvidenceMetric(label="Machine-failure rate", value=f"{rate:.2%}", source_tool="get_failure_rate"),
            ]
        )

    ranges = results.get("failure_rate_by_range")
    if ranges:
        range_rows = ranges["ranges"]
        for row in range_rows:
            value = "not estimable" if row["failure_rate"] is None else f"{row['failure_rate']:.2%}"
            metrics.append(EvidenceMetric(label=f"Failure rate: {row['label']}", value=value, source_tool="failure_rate_by_range"))
        if len(range_rows) >= 2:
            lower, higher = range_rows[0], range_rows[-1]
            if lower["failure_rate"] is not None and higher["failure_rate"] is not None:
                direction = "higher" if higher["failure_rate"] > lower["failure_rate"] else "not higher"
                findings.append(
                    EvidenceFinding(
                        statement=(
                            f"The highest planned range ({higher['label']}) has a {higher['failure_rate']:.2%} "
                            f"failure rate versus {lower['failure_rate']:.2%} in {lower['label']}; it is {direction} "
                            "in this selected data."
                        ),
                        source_tools=["failure_rate_by_range"],
                    )
                )

    products = results.get("compare_product_types")
    if products:
        for group in products["groups"]:
            metrics.append(EvidenceMetric(label=f"{group['product_type']} failure rate", value=f"{group['failure_rate']:.2%}", source_tool="compare_product_types"))
        findings.append(EvidenceFinding(statement="Product types are compared using selected-observation failure rates.", source_tools=["compare_product_types"]))

    comparison = results.get("compare_failed_vs_healthy")
    if comparison:
        for variable, groups in comparison["variables"].items():
            failed_mean = groups["failed"]["mean"]
            healthy_mean = groups["healthy"]["mean"]
            if failed_mean is not None and healthy_mean is not None:
                findings.append(EvidenceFinding(statement=f"Mean {variable}: failed {failed_mean:.2f}, healthy {healthy_mean:.2f}.", source_tools=["compare_failed_vs_healthy"]))

    conditional = results.get("analyze_conditional_relationship")
    if conditional:
        statistical_tests.append(conditional)
        findings.append(
            EvidenceFinding(
                statement=conditional["interpretation"],
                source_tools=["analyze_conditional_relationship"],
            )
        )
        uncertainty.append("Conditional-analysis estimates are associations, not evidence that changing one variable causes failure.")

    modes = results.get("failure_mode_breakdown")
    if modes:
        engineering_evidence.append(modes)
        findings.append(EvidenceFinding(statement=modes["note"], source_tools=["failure_mode_breakdown"]))

    regimes = results.get("discover_high_risk_regimes")
    if regimes:
        engineering_evidence.append(regimes)
        uncertainty.append("Risk regimes are internally evaluated on a held-out split, but remain dataset-specific.")

    audit = results.get("run_data_contract_audit")
    if audit:
        warnings.extend(issue["message"] for issue in audit.get("issues", []))
        findings.append(EvidenceFinding(statement=f"Data contract audit: {audit['row_count']} observations checked.", source_tools=["run_data_contract_audit"]))

    risk = results.get("get_model_risk")
    if risk:
        model_evidence.append(risk)
        metrics.append(EvidenceMetric(label="Calibrated model risk", value=f"{risk['risk_probability']:.2%}", source_tool="get_model_risk"))
        uncertainty.append("The calibrated probability is a model estimate, not a diagnosis or maintenance instruction.")

    if plan.intent == "unavailable_data":
        warnings.append(plan.clarification or "The requested data are unavailable in AI4I.")
    if plan.intent == "unsupported_question":
        warnings.append(plan.clarification or "The question is outside this copilot's supported analytical scope.")

    return EvidencePackage(
        question=question,
        intent=plan.intent,
        filters=filters,
        sample_size=sample_size,
        calculations_run=list(results),
        findings=findings,
        metrics=metrics,
        visualisations=_visualisations_for(plan.intent, results),
        statistical_tests=statistical_tests,
        model_evidence=model_evidence,
        engineering_evidence=engineering_evidence,
        data_quality_warnings=warnings,
        uncertainty=uncertainty,
        limitations=LIMITATIONS,
        suggested_next_questions=_next_questions(plan.intent),
        tool_results=results,
    )


def _sample_size(results: dict[str, Any]) -> int | None:
    for result in results.values():
        for key in ("observation_count", "selected_observation_count", "candidate_count"):
            if key in result:
                return result[key]
    return None


def _visualisations_for(intent: str, results: dict[str, Any]) -> list[str]:
    mapping = {
        "failure_rate": "RPM failure-rate chart",
        "failed_healthy_comparison": "Failed-versus-healthy distribution chart",
        "product_comparison": "Product-type comparison chart",
        "failure_investigation": "RPM–torque failure map and HDF envelope",
        "similar_conditions": "Similar-conditions comparison chart",
    }
    return [mapping[intent]] if intent in mapping and results else []


def _next_questions(intent: str) -> list[str]:
    if intent == "failure_rate":
        return ["Compare failed and healthy torque.", "Show failure rate by RPM."]
    if intent == "failure_investigation":
        return ["Compare failed and healthy torque.", "Show the high-RPM failure-mode breakdown."]
    return ["What percentage failed?", "Compare L, M, and H products."]
