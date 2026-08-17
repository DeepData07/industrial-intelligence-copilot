"""Replay immutable AI4I observations as honest simulated live telemetry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    PRODUCT_ID,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
    UID,
    assert_expected_columns,
)
from industrial_copilot.simulation.schemas import ReplaySpeed, SimulationSession, TelemetryEvent

AI4I_REPLAY_LABEL = "AI4I Dataset Replay"


class AI4IReplayEngine:
    """Stateful replay controller with start/pause/resume/reset and end-of-data handling."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        asset_id: str = "MACHINE-01",
        rolling_window_size: int = 50,
        start_time: datetime | None = None,
    ) -> None:
        assert_expected_columns(frame)
        if rolling_window_size < 1:
            raise ValueError("rolling_window_size must be at least 1.")
        self._frame = frame.copy(deep=True).reset_index(drop=True)
        self._asset_id = asset_id
        self._rolling_window_size = rolling_window_size
        self._base_start_time = start_time or datetime(2026, 8, 15, 0, 0, tzinfo=UTC)
        self._session_counter = 1
        self._session = self._new_session(status="idle")

    @property
    def session(self) -> SimulationSession:
        """Return the current immutable session snapshot."""

        return self._session

    def start(self, *, speed: ReplaySpeed | None = None) -> SimulationSession:
        """Start an idle replay or restart from a completed session with a new session ID."""

        if self._session.status == "complete":
            self.reset()
        if speed is not None:
            self.set_speed(speed)
        self._session = self._session.model_copy(update={"status": "running"})
        return self._session

    def pause(self) -> SimulationSession:
        """Pause a running replay without clearing the rolling history."""

        if self._session.status == "running":
            self._session = self._session.model_copy(update={"status": "paused"})
        return self._session

    def resume(self) -> SimulationSession:
        """Resume a paused replay."""

        if self._session.status == "paused":
            self._session = self._session.model_copy(update={"status": "running"})
        return self._session

    def reset(self) -> SimulationSession:
        """Create a fresh visible simulation session and clear replay state."""

        self._session_counter += 1
        self._session = self._new_session(status="idle")
        return self._session

    def set_speed(self, speed: ReplaySpeed) -> SimulationSession:
        """Set replay speed metadata used by next_events_for_tick."""

        if speed not in (1, 5, 10, 20):
            raise ValueError("Replay speed must be one of: 1, 5, 10, 20.")
        self._session = self._session.model_copy(update={"speed": speed})
        return self._session

    def next_event(self) -> TelemetryEvent | None:
        """Emit one telemetry event if running; mark complete at end of the dataset."""

        if self._session.status != "running":
            return None
        if self._session.cursor_index >= len(self._frame):
            self._session = self._session.model_copy(update={"status": "complete"})
            return None

        event = self._event_from_row(
            self._frame.iloc[self._session.cursor_index],
            cycle_id=self._session.cursor_index + 1,
        )
        history = (*self._session.history, event)[-self._rolling_window_size :]
        next_cursor = self._session.cursor_index + 1
        next_status = "complete" if next_cursor >= len(self._frame) else "running"
        self._session = self._session.model_copy(
            update={"cursor_index": next_cursor, "status": next_status, "history": history}
        )
        return event

    def next_events_for_tick(self) -> tuple[TelemetryEvent, ...]:
        """Emit up to speed events for one UI refresh tick."""

        events: list[TelemetryEvent] = []
        for _ in range(self._session.speed):
            event = self.next_event()
            if event is None:
                break
            events.append(event)
            if self._session.status == "complete":
                break
        return tuple(events)

    def _new_session(self, *, status: str) -> SimulationSession:
        return SimulationSession(
            session_id=f"SIM-{self._session_counter:04d}",
            asset_id=self._asset_id,
            source="ai4i_replay",
            source_label=AI4I_REPLAY_LABEL,
            status=status,
            total_cycles=len(self._frame),
            rolling_window_size=self._rolling_window_size,
        )

    def _event_from_row(self, row: pd.Series, *, cycle_id: int) -> TelemetryEvent:
        return TelemetryEvent(
            simulation_session_id=self._session.session_id,
            asset_id=self._asset_id,
            cycle_id=cycle_id,
            simulated_timestamp=self._base_start_time + timedelta(seconds=cycle_id - 1),
            source="ai4i_replay",
            source_label=AI4I_REPLAY_LABEL,
            uid=int(row[UID]),
            product_id=str(row[PRODUCT_ID]),
            product_type=str(row[PRODUCT_TYPE]),
            air_temperature_k=float(row[AIR_TEMPERATURE]),
            process_temperature_k=float(row[PROCESS_TEMPERATURE]),
            rotational_speed_rpm=float(row[ROTATIONAL_SPEED]),
            torque_nm=float(row[TORQUE]),
            tool_wear_min=float(row[TOOL_WEAR]),
        )
