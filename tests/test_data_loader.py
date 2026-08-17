from __future__ import annotations

import pandas as pd
import pytest

from industrial_copilot.data.loader import load_ai4i_data
from industrial_copilot.data.schema import (
    EXPECTED_COLUMNS,
    SOURCE_UID_ALIAS,
    UID,
    DataContractError,
)


def test_loader_rejects_missing_required_column(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(columns=EXPECTED_COLUMNS[:-1]).to_csv(path, index=False)

    with pytest.raises(DataContractError, match="missing columns"):
        load_ai4i_data(path)


def test_loader_does_not_mutate_raw_file(tmp_path) -> None:
    path = tmp_path / "source.csv"
    pd.DataFrame([[1] * len(EXPECTED_COLUMNS)], columns=EXPECTED_COLUMNS).to_csv(path, index=False)
    before = path.read_bytes()

    frame = load_ai4i_data(path)

    assert list(frame.columns) == list(EXPECTED_COLUMNS)
    assert path.read_bytes() == before


def test_loader_maps_published_udi_to_canonical_uid_in_memory(tmp_path) -> None:
    path = tmp_path / "uci_source.csv"
    source_columns = [SOURCE_UID_ALIAS if column == UID else column for column in EXPECTED_COLUMNS]
    pd.DataFrame([[1] * len(source_columns)], columns=source_columns).to_csv(path, index=False)

    frame = load_ai4i_data(path)

    assert UID in frame.columns
    assert SOURCE_UID_ALIAS not in frame.columns
    assert SOURCE_UID_ALIAS in path.read_text(encoding="utf-8").splitlines()[0]
