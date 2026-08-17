"""Failure-mode evidence tools that preserve overlapping labels."""

from __future__ import annotations

import pandas as pd

from industrial_copilot.analytics.common import (
    apply_filters,
    failure_rate,
    prepare_analysis_frame,
    resolve_filters,
)
from industrial_copilot.analytics.models import (
    AnalysisFilters,
    FailureModeBreakdown,
    FailureModeBreakdownItem,
)
from industrial_copilot.data.schema import FAILURE_MODES


def failure_mode_breakdown(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> FailureModeBreakdown:
    """Report flag prevalence without incorrectly treating possibly overlapping modes as exclusive."""

    resolved = resolve_filters(filters)
    selected = apply_filters(prepare_analysis_frame(frame), resolved)
    observation_count, failed_count, _ = failure_rate(selected)
    modes = []
    for mode in FAILURE_MODES:
        flagged_count = int(selected[mode].eq(1).sum())
        modes.append(
            FailureModeBreakdownItem(
                failure_mode=mode,
                flagged_observation_count=flagged_count,
                rate_among_selected=flagged_count / observation_count if observation_count else 0.0,
                share_of_failed_observations=flagged_count / failed_count if failed_count else None,
            )
        )
    return FailureModeBreakdown(
        selected_observation_count=observation_count,
        selected_failed_observation_count=failed_count,
        modes=modes,
        filters=resolved,
        note="Failure-mode flags can overlap and do not necessarily sum to Machine failure.",
    )
