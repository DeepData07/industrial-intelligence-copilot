"""Download the published AI4I CSV once, keeping the raw file immutable afterwards."""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
UCI_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)


def main() -> int:
    """Fetch and extract the UCI archive without overwriting an existing raw source file."""

    if DESTINATION.exists():
        digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
        print(f"Raw dataset already exists: {DESTINATION}")
        print(f"SHA256: {digest}")
        print("No overwrite performed; raw data is immutable.")
        return 0

    try:
        with urlopen(UCI_ARCHIVE_URL, timeout=30) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise RuntimeError(f"Expected one CSV in UCI archive, found: {csv_members}")
            csv_bytes = archive.read(csv_members[0])
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(csv_bytes)
    print(f"Downloaded raw dataset to: {DESTINATION}")
    print(f"Rows will be validated by the data contract; SHA256: {hashlib.sha256(csv_bytes).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
