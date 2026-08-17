from __future__ import annotations

import joblib
import pandas as pd
import pytest

from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    EXPECTED_COLUMNS,
    PROCESS_TEMPERATURE,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
)
from industrial_copilot.features.engineering import MECHANICAL_POWER, calculate_operating_features
from industrial_copilot.ml.predict import (
    get_model_risk,
    load_fitted_risk_model,
    risk_level_for_probability,
)
from industrial_copilot.ml.schemas import PredictionInput
from industrial_copilot.ml.train import (
    ENGINEERING_AUGMENTED_FEATURES,
    LEAKAGE_EXCLUDED_COLUMNS,
    RAW_FEATURES,
    train_all_models,
)


def synthetic_training_frame() -> pd.DataFrame:
    """Build a sufficiently sized, deterministic source-shaped training frame for pipeline tests."""

    rows = []
    for index in range(1, 181):
        product_type = ("L", "M", "H")[index % 3]
        rpm = 1200 + (index * 29) % 1200
        torque = 18.0 + (index * 7) % 55
        wear = (index * 13) % 240
        failure = int((torque >= 55 and wear >= 150) or (rpm < 1380 and index % 5 == 0))
        rows.append(
            [
                index,
                f"{product_type}{index:05d}",
                product_type,
                300.0 + (index % 10) / 10,
                310.0 + (index % 10) / 10,
                rpm,
                torque,
                wear,
                failure,
                0,
                0,
                0,
                0,
                0,
            ]
        )
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


@pytest.fixture(scope="module")
def trained_models():
    return train_all_models(synthetic_training_frame(), test_fraction=0.25, random_state=7)


def test_feature_sets_exclude_every_target_and_failure_mode() -> None:
    assert not (set(RAW_FEATURES) & set(LEAKAGE_EXCLUDED_COLUMNS))
    assert not (set(ENGINEERING_AUGMENTED_FEATURES) & set(LEAKAGE_EXCLUDED_COLUMNS))


def test_operating_features_work_without_labels() -> None:
    telemetry = pd.DataFrame(
        [
            {
                PRODUCT_TYPE: "L",
                AIR_TEMPERATURE: 300.0,
                PROCESS_TEMPERATURE: 310.0,
                ROTATIONAL_SPEED: 1500.0,
                TORQUE: 40.0,
                TOOL_WEAR: 100.0,
            }
        ]
    )

    engineered = calculate_operating_features(telemetry)

    assert MECHANICAL_POWER in engineered.columns
    assert "Machine failure" not in engineered.columns


def test_training_reports_all_required_models_and_metrics(trained_models) -> None:
    result, models = trained_models

    assert len(result.results) == 4
    assert len(models) == 4
    for evaluation in result.results:
        assert evaluation.calibrated.pr_auc is not None
        assert evaluation.calibrated.roc_auc is not None
        assert evaluation.calibrated.brier_score >= 0
        assert evaluation.calibrated.calibration_curve
        assert not (set(evaluation.input_features) & set(LEAKAGE_EXCLUDED_COLUMNS))


def test_prediction_uses_calibrated_artifact_without_labels(tmp_path, trained_models) -> None:
    _, models = trained_models
    model = models["logistic_regression__engineering_augmented"]
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = load_fitted_risk_model(path)

    prediction = get_model_risk(
        loaded,
        PredictionInput(
            product_type="L",
            air_temperature_k=300.0,
            process_temperature_k=308.0,
            rotational_speed_rpm=1300.0,
            torque_nm=55.0,
            tool_wear_min=200.0,
        ),
    )

    assert 0 <= prediction.failure_probability <= 1
    assert prediction.risk_level in {"LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "REVIEW_REQUIRED"}


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.01, "LOW_RISK"), (0.10, "MEDIUM_RISK"), (0.20, "HIGH_RISK"), (0.50, "REVIEW_REQUIRED")],
)
def test_risk_bands_are_cautious_and_deterministic(probability: float, expected: str) -> None:
    assert risk_level_for_probability(probability) == expected
