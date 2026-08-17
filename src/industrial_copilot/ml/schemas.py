"""Typed training, evaluation, calibration, and prediction outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["logistic_regression", "random_forest"]
FeatureSetName = Literal["raw", "engineering_augmented"]
RiskLevel = Literal["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "REVIEW_REQUIRED"]


class ConfusionMatrix(BaseModel):
    """Threshold-0.5 confusion matrix for an explicitly held-out test set."""

    model_config = ConfigDict(frozen=True)

    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_positive: int = Field(ge=0)


class CalibrationPoint(BaseModel):
    """Observed versus predicted failure frequency in a probability bin."""

    model_config = ConfigDict(frozen=True)

    bin_lower: float = Field(ge=0, le=1)
    bin_upper: float = Field(ge=0, le=1)
    observation_count: int = Field(ge=1)
    mean_predicted_probability: float = Field(ge=0, le=1)
    observed_failure_rate: float = Field(ge=0, le=1)


class EvaluationMetrics(BaseModel):
    """Imbalance-aware held-out evaluation metrics and calibration evidence."""

    model_config = ConfigDict(frozen=True)

    pr_auc: float | None = Field(default=None, ge=0, le=1)
    roc_auc: float | None = Field(default=None, ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    brier_score: float = Field(ge=0, le=1)
    confusion_matrix: ConfusionMatrix
    calibration_curve: list[CalibrationPoint]
    threshold: float = Field(default=0.5, gt=0, lt=1)


class ModelEvaluationResult(BaseModel):
    """One trained model's held-out metrics before and after probability calibration."""

    model_config = ConfigDict(frozen=True)

    model_name: ModelName
    feature_set: FeatureSetName
    input_features: list[str]
    train_observation_count: int = Field(ge=1)
    test_observation_count: int = Field(ge=1)
    train_failure_rate: float = Field(ge=0, le=1)
    test_failure_rate: float = Field(ge=0, le=1)
    uncalibrated: EvaluationMetrics
    calibrated: EvaluationMetrics
    calibration_method: str
    note: str


class TrainingRunResult(BaseModel):
    """Summary of all required model/feature-set combinations."""

    model_config = ConfigDict(frozen=True)

    target: str
    random_state: int
    test_fraction: float = Field(gt=0, lt=1)
    results: list[ModelEvaluationResult]
    leakage_excluded_columns: list[str]
    note: str


class PredictionInput(BaseModel):
    """Telemetry available at prediction time; labels are deliberately absent."""

    model_config = ConfigDict(frozen=True)

    product_type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)


class RiskPrediction(BaseModel):
    """Calibrated model probability with an intentionally non-certain risk label."""

    model_config = ConfigDict(frozen=True)

    model_name: ModelName
    feature_set: FeatureSetName
    failure_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    note: str
