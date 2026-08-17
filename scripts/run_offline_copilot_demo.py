"""Run a reproducible copilot demonstration without an external LLM."""

from __future__ import annotations

import sys

from industrial_copilot.copilot.service import IndustrialCopilotService
from industrial_copilot.copilot.state import ConversationState


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    service = IndustrialCopilotService()
    state = ConversationState()
    for question in (
        "What percentage failed?",
        "Only L products.",
        "Now compare torque.",
        "Why are failures higher at high RPM?",
        "What happened during the last 30 days?",
    ):
        response = service.ask(question, state)
        state = ConversationState.model_validate(response.state)
        print(f"Question: {question}")
        print(f"Intent: {response.evidence.intent}")
        print(f"Answer: {response.answer}")
        print(f"Tools: {', '.join(response.evidence.calculations_run) or 'none'}")
        print()


if __name__ == "__main__":
    main()
