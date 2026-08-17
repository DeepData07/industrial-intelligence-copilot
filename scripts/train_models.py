"""Train and persist calibrated models on the local AI4I data."""

from __future__ import annotations

from industrial_copilot.config import PROJECT_ROOT
from industrial_copilot.data.loader import load_ai4i_data
from industrial_copilot.ml.train import train_and_save_models


def main() -> None:
    """Write local ignored model artifacts and print compact held-out metrics."""

    result = train_and_save_models(load_ai4i_data(), PROJECT_ROOT / "models")
    print(f"Saved {len(result.results)} calibrated model artifacts to: {PROJECT_ROOT / 'models'}")
    for evaluation in result.results:
        metrics = evaluation.calibrated
        print(
            f"{evaluation.model_name} / {evaluation.feature_set}: "
            f"PR-AUC={metrics.pr_auc:.3f}, ROC-AUC={metrics.roc_auc:.3f}, "
            f"Brier={metrics.brier_score:.4f}"
        )


if __name__ == "__main__":
    main()
