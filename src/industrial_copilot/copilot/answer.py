"""Short deterministic renderers for evidence-first offline answers."""

from __future__ import annotations

from industrial_copilot.copilot.schemas import EvidencePackage, ToolPlan


def render_offline_answer(plan: ToolPlan, evidence: EvidencePackage) -> str:
    """Render only claims already represented in the evidence package."""

    if plan.clarification:
        return plan.clarification
    if evidence.findings:
        return " ".join(finding.statement for finding in evidence.findings[:3])
    if evidence.metrics:
        return "; ".join(f"{metric.label}: {metric.value}" for metric in evidence.metrics[:3]) + "."
    return "No result was produced for the selected AI4I subset."
