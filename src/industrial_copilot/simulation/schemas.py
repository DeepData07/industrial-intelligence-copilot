"""Typed event and session contracts for simulated live operations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReplaySpeed = Literal[1, 5, 10, 20]
SimulationSource = Literal["ai4i_replay", "synthetic_demo_scenario"]
SimulationStatus = Literal["idle", "running", "paused", "complete"]


class TelemetryEvent(BaseModel):
    """One simulated telemetry cycle with explicit source honesty fields."""

    model_config = ConfigDict(frozen=True)

    simulation_session_id: str
    asset_id: str
    cycle_id: int = Field(ge=1)
    simulated_timestamp: datetime
    source: SimulationSource
    source_label: str
    uid: int | None = Field(default=None, ge=1)
    product_id: str | None = None
    product_type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)


class SimulationSession(BaseModel):
    """Current replay session state plus a bounded rolling telemetry history."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    asset_id: str
    source: SimulationSource
    source_label: str
    status: SimulationStatus = "idle"
    speed: ReplaySpeed = 1
    cursor_index: int = Field(default=0, ge=0)
    total_cycles: int = Field(ge=0)
    rolling_window_size: int = Field(default=50, ge=1)
    history: tuple[TelemetryEvent, ...] = ()

    @property
    def current_cycle(self) -> int:
        """Return how many source observations have been emitted in this session."""

        return self.cursor_index

