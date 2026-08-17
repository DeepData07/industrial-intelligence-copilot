from __future__ import annotations

import pandas as pd
import pytest

from industrial_copilot.data.schema import EXPECTED_COLUMNS


@pytest.fixture
def sample_ai4i_frame() -> pd.DataFrame:
    """Small AI4I-shaped frame spanning healthy and documented failure conditions."""

    return pd.DataFrame(
        [
            [1, "L00001", "L", 300.0, 310.0, 1500, 40.0, 10, 0, 0, 0, 0, 0, 0],
            [2, "M00002", "M", 300.0, 308.0, 1300, 40.0, 10, 1, 0, 1, 0, 0, 0],
            [3, "H00003", "H", 300.0, 310.0, 2000, 10.0, 10, 1, 0, 0, 1, 0, 0],
            [4, "L00004", "L", 300.0, 310.0, 1500, 50.0, 230, 1, 0, 0, 0, 1, 0],
            [5, "L00005", "L", 300.0, 310.0, 1700, 40.0, 220, 1, 1, 0, 0, 0, 0],
        ],
        columns=EXPECTED_COLUMNS,
    )
