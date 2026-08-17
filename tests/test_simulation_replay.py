from __future__ import annotations

import pandas as pd
import pytest

from industrial_copilot.data.schema import TORQUE
from industrial_copilot.simulation.replay import AI4I_REPLAY_LABEL, AI4IReplayEngine


def test_ai4i_replay_emits_source_honest_events_without_mutating_source(sample_ai4i_frame) -> None:
    before = sample_ai4i_frame.copy(deep=True)
    engine = AI4IReplayEngine(sample_ai4i_frame, rolling_window_size=3)

    engine.start(speed=1)
    event = engine.next_event()

    assert event is not None
    assert event.simulation_session_id == "SIM-0001"
    assert event.asset_id == "MACHINE-01"
    assert event.cycle_id == 1
    assert event.source == "ai4i_replay"
    assert event.source_label == AI4I_REPLAY_LABEL
    assert event.uid == 1
    assert event.torque_nm == pytest.approx(float(sample_ai4i_frame.loc[0, TORQUE]))
    pd.testing.assert_frame_equal(sample_ai4i_frame, before)


def test_replay_pause_resume_reset_and_end_of_dataset_behaviour(sample_ai4i_frame) -> None:
    engine = AI4IReplayEngine(sample_ai4i_frame, rolling_window_size=2)

    assert engine.session.status == "idle"
    engine.start(speed=5)
    assert engine.session.status == "running"
    assert len(engine.next_events_for_tick()) == len(sample_ai4i_frame)
    assert engine.session.status == "complete"
    assert engine.session.current_cycle == len(sample_ai4i_frame)
    assert len(engine.session.history) == 2

    old_session_id = engine.session.session_id
    engine.start()
    assert engine.session.session_id != old_session_id
    assert engine.session.status == "running"
    assert engine.session.current_cycle == 0

    engine.pause()
    assert engine.session.status == "paused"
    assert engine.next_event() is None
    engine.resume()
    assert engine.session.status == "running"
    assert engine.next_event() is not None

    reset_session = engine.reset()
    assert reset_session.status == "idle"
    assert reset_session.current_cycle == 0
    assert reset_session.history == ()
    assert reset_session.session_id != old_session_id


def test_replay_rejects_invalid_inputs(sample_ai4i_frame) -> None:
    with pytest.raises(ValueError, match="rolling_window_size"):
        AI4IReplayEngine(sample_ai4i_frame, rolling_window_size=0)

    engine = AI4IReplayEngine(sample_ai4i_frame)
    with pytest.raises(ValueError, match="Replay speed"):
        engine.set_speed(2)  # type: ignore[arg-type]
