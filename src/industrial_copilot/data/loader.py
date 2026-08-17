"""Read raw AI4I data without mutating or cleaning source observations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from industrial_copilot.config import get_settings
from industrial_copilot.data.schema import (
    SOURCE_UID_ALIAS,
    UID,
    DataContractError,
    assert_expected_columns,
)


def load_ai4i_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the exact raw CSV after a schema check; no cleaning or imputation occurs."""

    source_path = Path(path) if path is not None else get_settings().raw_data_path
    if not source_path.exists():
        raise FileNotFoundError(
            f"AI4I raw data was not found at {source_path}. "
            "Run `python scripts/download_data.py` first."
        )

    try:
        frame = pd.read_csv(source_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DataContractError(f"Unable to read AI4I CSV at {source_path}: {error}") from error

    # UCI publishes this identifier as "UDI". The application uses the
    # conventional name "UID" internally without modifying the source file.
    if SOURCE_UID_ALIAS in frame.columns and UID not in frame.columns:
        frame = frame.rename(columns={SOURCE_UID_ALIAS: UID})

    assert_expected_columns(frame)
    return frame
