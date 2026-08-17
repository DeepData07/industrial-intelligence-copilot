# Data policy

`data/raw/` is reserved for downloaded, source-format AI4I files and is treated as immutable input. The application must never modify, impute, trim, winsorize, or overwrite raw data.

Run `python scripts/download_data.py` from the repository root to download the published `ai4i2020.csv` from the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset). The script refuses to overwrite an existing file and prints its SHA-256 hash.

UCI's source header spells the observation identifier `UDI`; the application uses the conventional name `UID`. The loader applies this mapping in memory after reading the file. The raw CSV is never changed.

Run `python scripts/run_data_audit.py` to produce a JSON report for schema, missing values, duplicates, data types, physical sanity checks, label distributions, and documented-rule agreement. The report distinguishes `Machine failure = 1` with no active failure-mode flag from a failure-mode flag where `Machine failure = 0`; it reports these as source warnings and never rewrites them.

`data/processed/` is reserved for reproducible derived data only. The current pipeline does not write derived data there.

The dataset is created by Stephan Matzka and published by the UCI Machine Learning Repository under CC BY 4.0. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
