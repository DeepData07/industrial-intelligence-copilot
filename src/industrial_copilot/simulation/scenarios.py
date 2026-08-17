"""Curated demo telemetry scenarios based on documented AI4I operating rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

import pandas as pd

from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
)
from industrial_copilot.simulation.schemas import TelemetryEvent

ScenarioName = Literal["osf", "hdf", "pwf"]
SYNTHETIC_SCENARIO_LABEL = "Synthetic Demo Scenario Based on AI4I Rules"


def generate_scenario_events(
    scenario_name: ScenarioName,
    *,
    session_id: str = "SIM-SCENARIO-001",
    asset_id: str = "MACHINE-01",
    cycles: int = 36,
    start_time: datetime | None = None,
) -> tuple[TelemetryEvent, ...]:
    """Generate one supported scenario by name."""

    if scenario_name == "osf":
        return generate_osf_scenario(
            session_id=session_id,
            asset_id=asset_id,
            cycles=cycles,
            start_time=start_time,
        )
    if scenario_name == "hdf":
        return generate_hdf_scenario(
            session_id=session_id,
            asset_id=asset_id,
            cycles=cycles,
            start_time=start_time,
        )
    if scenario_name == "pwf":
        return generate_pwf_scenario(
            session_id=session_id,
            asset_id=asset_id,
            cycles=cycles,
            start_time=start_time,
        )
    raise ValueError(f"Unsupported scenario: {scenario_name}")


def generate_osf_scenario(
    *,
    session_id: str = "SIM-OSF-001",
    asset_id: str = "MACHINE-01",
    cycles: int = 36,
    start_time: datetime | None = None,
) -> tuple[TelemetryEvent, ...]:
    """Move torque and tool wear toward the documented L-product OSF threshold."""

    return _generate_linear_scenario(
        session_id=session_id,
        asset_id=asset_id,
        cycles=cycles,
        start_time=start_time,
        product_type="L",
        air_temperature_start=298.0,
        air_temperature_end=298.4,
        process_temperature_start=308.0,
        process_temperature_end=308.5,
        rotational_speed_start=1450.0,
        rotational_speed_end=1440.0,
        torque_start=42.0,
        torque_end=56.0,
        tool_wear_start=170.0,
        tool_wear_end=218.0,
    )


def generate_hdf_scenario(
    *,
    session_id: str = "SIM-HDF-001",
    asset_id: str = "MACHINE-01",
    cycles: int = 36,
    start_time: datetime | None = None,
) -> tuple[TelemetryEvent, ...]:
    """Move temperature delta and RPM toward the documented HDF envelope."""

    return _generate_linear_scenario(
        session_id=session_id,
        asset_id=asset_id,
        cycles=cycles,
        start_time=start_time,
        product_type="M",
        air_temperature_start=298.0,
        air_temperature_end=300.0,
        process_temperature_start=308.0,
        process_temperature_end=307.9,
        rotational_speed_start=1500.0,
        rotational_speed_end=1320.0,
        torque_start=39.0,
        torque_end=41.0,
        tool_wear_start=80.0,
        tool_wear_end=100.0,
    )


def generate_pwf_scenario(
    *,
    session_id: str = "SIM-PWF-001",
    asset_id: str = "MACHINE-01",
    cycles: int = 36,
    start_time: datetime | None = None,
) -> tuple[TelemetryEvent, ...]:
    """Move RPM and torque toward the documented high-power PWF region."""

    return _generate_linear_scenario(
        session_id=session_id,
        asset_id=asset_id,
        cycles=cycles,
        start_time=start_time,
        product_type="H",
        air_temperature_start=298.5,
        air_temperature_end=299.0,
        process_temperature_start=308.7,
        process_temperature_end=309.0,
        rotational_speed_start=1500.0,
        rotational_speed_end=2500.0,
        torque_start=40.0,
        torque_end=36.5,
        tool_wear_start=60.0,
        tool_wear_end=80.0,
    )


def events_to_operating_frame(events: tuple[TelemetryEvent, ...]) -> pd.DataFrame:
    """Convert telemetry events into the operating-input shape used by feature engineering."""

    return pd.DataFrame(
        [
            {
                PRODUCT_TYPE: event.product_type,
                AIR_TEMPERATURE: event.air_temperature_k,
                PROCESS_TEMPERATURE: event.process_temperature_k,
                ROTATIONAL_SPEED: event.rotational_speed_rpm,
                TORQUE: event.torque_nm,
                TOOL_WEAR: event.tool_wear_min,
            }
            for event in events
        ]
    )


def _generate_linear_scenario(
    *,
    session_id: str,
    asset_id: str,
    cycles: int,
    start_time: datetime | None,
    product_type: Literal["L", "M", "H"],
    air_temperature_start: float,
    air_temperature_end: float,
    process_temperature_start: float,
    process_temperature_end: float,
    rotational_speed_start: float,
    rotational_speed_end: float,
    torque_start: float,
    torque_end: float,
    tool_wear_start: float,
    tool_wear_end: float,
) -> tuple[TelemetryEvent, ...]:
    if cycles < 2:
        raise ValueError("Scenario cycles must be at least 2.")
    base_time = start_time or datetime(2026, 8, 15, 0, 0, tzinfo=UTC)
    events: list[TelemetryEvent] = []
    for index in range(cycles):
        fraction = index / (cycles - 1)
        events.append(
            TelemetryEvent(
                simulation_session_id=session_id,
                asset_id=asset_id,
                cycle_id=index + 1,
                simulated_timestamp=base_time + timedelta(seconds=index),
                source="synthetic_demo_scenario",
                source_label=SYNTHETIC_SCENARIO_LABEL,
                uid=None,
                product_id=None,
                product_type=product_type,
                air_temperature_k=_lerp(air_temperature_start, air_temperature_end, fraction),
                process_temperature_k=_lerp(
                    process_temperature_start,
                    process_temperature_end,
                    fraction,
                ),
                rotational_speed_rpm=_lerp(
                    rotational_speed_start,
                    rotational_speed_end,
                    fraction,
                ),
                torque_nm=_lerp(torque_start, torque_end, fraction),
                tool_wear_min=_lerp(tool_wear_start, tool_wear_end, fraction),
            )
        )
    return tuple(events)


def _lerp(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction
