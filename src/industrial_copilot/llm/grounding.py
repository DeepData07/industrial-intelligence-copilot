"""Deterministic claim-ledger construction and validation for grounded AI answers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from industrial_copilot.copilot.schemas import (
    EvidenceAtom,
    EvidencePackage,
    GroundedCopilotAnswer,
)

NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
FORBIDDEN_LANGUAGE = (
    "will fail",
    "guaranteed root cause",
    "definitely caused",
    "issue a plc command",
    "shut down the machine",
)


def build_claim_ledger(
    evidence: EvidencePackage,
    *,
    scenario_id: str | None = None,
    cycle: int | None = None,
    incident_id: str | None = None,
) -> list[EvidenceAtom]:
    """Build facts solely from deterministic evidence; no LLM call is involved."""

    atoms: list[EvidenceAtom] = []
    for index, metric in enumerate(evidence.metrics, start=1):
        atoms.append(
            EvidenceAtom(
                id=f"M{index}", kind="metric", statement=metric.label,
                display_value=metric.value, source=metric.source_tool,
                scenario_id=scenario_id, cycle=cycle, incident_id=incident_id,
            )
        )
    for index, finding in enumerate(evidence.findings, start=1):
        atoms.append(
            EvidenceAtom(
                id=f"F{index}", kind="finding", statement=finding.statement,
                source=", ".join(finding.source_tools), scenario_id=scenario_id,
                cycle=cycle, incident_id=incident_id,
            )
        )
    knowledge_index = 1
    for atom in evidence.knowledge_evidence:
        atoms.append(atom.model_copy(update={"id": f"K{knowledge_index}"}))
        knowledge_index += 1
    for index, limitation in enumerate([*evidence.limitations, *evidence.uncertainty], start=1):
        atoms.append(
            EvidenceAtom(
                id=f"L{index}", kind="limitation", statement=limitation,
                source="system_limit", authority="system_limit", scenario_id=scenario_id,
                cycle=cycle, incident_id=incident_id,
            )
        )
    return atoms


def validate_grounded_answer(answer: GroundedCopilotAnswer, ledger: Iterable[EvidenceAtom]) -> tuple[bool, str | None]:
    """Ensure every rendered statement cites known evidence and adds no unsupported numbers."""

    atoms = {atom.id: atom for atom in ledger}
    if not atoms:
        return False, "No claim ledger was available."
    statements = [answer.answer, *answer.evidence, *answer.next_checks, *answer.limitations]
    for statement in statements:
        if not statement.claim_ids or any(identifier not in atoms for identifier in statement.claim_ids):
            return False, "A response statement cited unknown evidence."
        if len(statement.text) > 900:
            return False, "A response statement was too long."
        lowered = statement.text.casefold()
        if any(item in lowered for item in FORBIDDEN_LANGUAGE):
            return False, "A response made an unsafe causal or machine-action claim."
        supported = _allowed_numbers(atoms[identifier] for identifier in statement.claim_ids)
        used = {_normalise_number(item) for item in NUMBER_RE.findall(statement.text)}
        if any(item not in supported for item in used):
            return False, "A response used an unsupported numeric value."
        if not _qualitatively_supported(statement.text, [atoms[item] for item in statement.claim_ids]):
            return False, "A response made an unsupported qualitative claim."
    return True, None


def repair_numeric_citations(
    answer: GroundedCopilotAnswer,
    ledger: Iterable[EvidenceAtom],
) -> GroundedCopilotAnswer:
    """Attach an existing evidence atom when a verified number was cited imprecisely.

    This only repairs references. It never changes prose, invents a value, or relaxes
    the grounding validator.
    """

    ledger_list = list(ledger)
    atoms = {atom.id: atom for atom in ledger_list}

    def repair(statement):
        claim_ids = [identifier for identifier in statement.claim_ids if identifier in atoms]
        cited_numbers = _allowed_numbers(atoms[identifier] for identifier in claim_ids)
        used_numbers = {_normalise_number(value) for value in NUMBER_RE.findall(statement.text)}
        for number in sorted(used_numbers - cited_numbers):
            supporting_id = next(
                (
                    atom.id
                    for atom in ledger_list
                    if number in _allowed_numbers([atom])
                ),
                None,
            )
            if supporting_id and supporting_id not in claim_ids:
                claim_ids.append(supporting_id)
        return statement.model_copy(update={"claim_ids": claim_ids})

    return answer.model_copy(
        update={
            "answer": repair(answer.answer),
            "evidence": [repair(item) for item in answer.evidence],
            "next_checks": [repair(item) for item in answer.next_checks],
            "limitations": [repair(item) for item in answer.limitations],
        }
    )


def _allowed_numbers(atoms: Iterable[EvidenceAtom]) -> set[str]:
    tokens: set[str] = set()
    for atom in atoms:
        for value in (atom.display_value, atom.statement, atom.value):
            if value is not None:
                for token in NUMBER_RE.findall(str(value)):
                    normalized = _normalise_number(token)
                    tokens.add(normalized)
                    if normalized.endswith(".0"):
                        tokens.add(normalized[:-2])
    return tokens


def _qualitatively_supported(text: str, atoms: list[EvidenceAtom]) -> bool:
    """A lightweight guard: statements must share meaningful terms with cited evidence."""

    ignored = {"the", "and", "with", "from", "this", "that", "current", "evidence", "is", "are", "to", "of", "a", "an", "for", "in", "on", "at", "it", "be", "can", "should"}
    terms = {term.casefold() for term in re.findall(r"[A-Za-z]{4,}", text) if term.casefold() not in ignored}
    source_terms = {
        term.casefold()
        for atom in atoms
        for term in re.findall(r"[A-Za-z]{4,}", f"{atom.statement} {atom.display_value or ''}")
    }
    return not terms or bool(terms & source_terms) or text.casefold().startswith(("inspect", "verify", "review", "consider"))


def _normalise_number(value: str) -> str:
    return value.replace(",", "").strip()
