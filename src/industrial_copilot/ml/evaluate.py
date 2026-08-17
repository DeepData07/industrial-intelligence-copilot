"""Held-out imbalanced-class evaluation and probability calibration evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from industrial_copilot.ml.schemas import CalibrationPoint, ConfusionMatrix, EvaluationMetrics


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> EvaluationMetrics:
    """Evaluate held-out probabilities without using accuracy as a primary metric."""

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")
    y = np.asarray(y_true, dtype=int)
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    roc_auc = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None
    pr_auc = float(average_precision_score(y, probabilities)) if len(np.unique(y)) == 2 else None
    return EvaluationMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        f1=float(f1_score(y, predicted, zero_division=0)),
        brier_score=float(brier_score_loss(y, probabilities)),
        confusion_matrix=ConfusionMatrix(
            true_negative=int(matrix[0, 0]),
            false_positive=int(matrix[0, 1]),
            false_negative=int(matrix[1, 0]),
            true_positive=int(matrix[1, 1]),
        ),
        calibration_curve=calibration_evidence(y, probabilities, calibration_bins),
        threshold=threshold,
    )


def calibration_evidence(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> list[CalibrationPoint]:
    """Return non-empty fixed probability bins with predicted and observed event frequency."""

    if bins < 2:
        raise ValueError("bins must be at least 2.")
    values = pd.DataFrame({"outcome": y_true, "probability": probabilities})
    edges = np.linspace(0, 1, bins + 1)
    values["bin"] = pd.cut(values["probability"], bins=edges, include_lowest=True)
    evidence: list[CalibrationPoint] = []
    for interval, group in values.groupby("bin", observed=True):
        evidence.append(
            CalibrationPoint(
                bin_lower=max(0.0, float(interval.left)),
                bin_upper=min(1.0, float(interval.right)),
                observation_count=len(group),
                mean_predicted_probability=float(group["probability"].mean()),
                observed_failure_rate=float(group["outcome"].mean()),
            )
        )
    return evidence
