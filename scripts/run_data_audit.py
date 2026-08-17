"""Load the raw AI4I CSV and print its structured data-contract audit as JSON."""

from __future__ import annotations

from industrial_copilot.data.audit import run_data_contract_audit
from industrial_copilot.data.loader import load_ai4i_data


def main() -> None:
    """Run an audit without writing to, cleaning, or transforming raw data."""

    audit = run_data_contract_audit(load_ai4i_data())
    print(audit.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
