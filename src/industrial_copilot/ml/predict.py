"""Calibrated risk prediction from telemetry available before a failure outcome is known."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
)
from industrial_copilot.features.engineering import calculate_operating_features
from industrial_copilot.ml.schemas import PredictionInput, RiskLevel, RiskPrediction
from industrial_copilot.ml.train import FittedRiskModel


def get_model_risk(model: FittedRiskModel, observation: PredictionInput) -> RiskPrediction:
    """Return a calibrated probability and a cautious risk band for one telemetry snapshot."""

    operating_frame = calculate_operating_features(_prediction_frame(observation))
    probability = float(model.estimator.predict_proba(operating_frame.loc[:, model.input_features])[:, 1][0])
    return RiskPrediction(
        model_name=model.model_name,
        feature_set=model.feature_set,
        failure_probability=probability,
        risk_level=risk_level_for_probability(probability),
        note=(
            "Calibrated probability is supporting evidence from a synthetic benchmark, not a "
            "certainty or a maintenance recommendation."
        ),
    )


def load_fitted_risk_model(path: Path | str) -> FittedRiskModel:
    """Load a locally persisted calibrated model artifact and validate its expected type."""

    model = joblib.load(path)
    if not isinstance(model, FittedRiskModel):
        raise TypeError("Artifact is not an Industrial Intelligence Copilot FittedRiskModel.")
    return model


def risk_level_for_probability(probability: float) -> RiskLevel:
    """Map probability to a cautious operational review band; thresholds are configurable later."""

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if probability < 0.05:
        return "LOW_RISK"
    if probability < 0.15:
        return "MEDIUM_RISK"
    if probability < 0.35:
        return "HIGH_RISK"
    return "REVIEW_REQUIRED"


def _prediction_frame(observation: PredictionInput) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                PRODUCT_TYPE: observation.product_type,
                AIR_TEMPERATURE: observation.air_temperature_k,
                PROCESS_TEMPERATURE: observation.process_temperature_k,
                ROTATIONAL_SPEED: observation.rotational_speed_rpm,
                TORQUE: observation.torque_nm,
                TOOL_WEAR: observation.tool_wear_min,
            }
        ]
    )
