"""Leakage-safe training for Logistic Regression and Random Forest risk evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from industrial_copilot.data.schema import (
    AIR_TEMPERATURE,
    FAILURE_MODES,
    MACHINE_FAILURE,
    PROCESS_TEMPERATURE,
    PRODUCT_TYPE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
)
from industrial_copilot.features.engineering import (
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
    TEMPERATURE_DELTA,
    calculate_engineering_features,
)
from industrial_copilot.ml.evaluate import evaluate_probabilities
from industrial_copilot.ml.schemas import (
    FeatureSetName,
    ModelEvaluationResult,
    ModelName,
    TrainingRunResult,
)

RAW_FEATURES = (
    PRODUCT_TYPE,
    AIR_TEMPERATURE,
    PROCESS_TEMPERATURE,
    ROTATIONAL_SPEED,
    TORQUE,
    TOOL_WEAR,
)
ENGINEERING_AUGMENTED_FEATURES = (
    *RAW_FEATURES,
    TEMPERATURE_DELTA,
    MECHANICAL_POWER,
    OVERSTRAIN_LOAD,
)
LEAKAGE_EXCLUDED_COLUMNS = (MACHINE_FAILURE, *FAILURE_MODES)


@dataclass
class FittedRiskModel:
    """Persistable calibrated classifier and the schema needed to construct its input frame."""

    model_name: ModelName
    feature_set: FeatureSetName
    input_features: tuple[str, ...]
    estimator: CalibratedClassifierCV


def train_all_models(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
    calibration_folds: int = 3,
) -> tuple[TrainingRunResult, dict[str, FittedRiskModel]]:
    """Fit and evaluate all required models using a stratified untouched test partition."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if calibration_folds < 2:
        raise ValueError("calibration_folds must be at least 2.")
    engineered = calculate_engineering_features(frame)
    target = engineered[MACHINE_FAILURE].astype(int)
    if set(target.unique()) - {0, 1}:
        raise ValueError("Machine failure target must contain only 0 and 1.")
    if target.value_counts().min() < calibration_folds + 1:
        raise ValueError("Insufficient minority-class observations for requested calibration folds.")

    train_index, test_index = train_test_split(
        engineered.index,
        test_size=test_fraction,
        random_state=random_state,
        stratify=target,
    )
    results: list[ModelEvaluationResult] = []
    fitted_models: dict[str, FittedRiskModel] = {}
    for feature_set in ("raw", "engineering_augmented"):
        features = features_for_set(feature_set)
        _assert_no_leakage(features)
        x_train, x_test = engineered.loc[train_index, features], engineered.loc[test_index, features]
        y_train, y_test = target.loc[train_index], target.loc[test_index]
        for model_name in ("logistic_regression", "random_forest"):
            base_estimator = build_pipeline(model_name, features, random_state)
            base_estimator.fit(x_train, y_train)
            uncalibrated = evaluate_probabilities(y_test, base_estimator.predict_proba(x_test)[:, 1])

            calibrated_estimator = CalibratedClassifierCV(
                estimator=build_pipeline(model_name, features, random_state),
                method="sigmoid",
                cv=calibration_folds,
            )
            calibrated_estimator.fit(x_train, y_train)
            calibrated = evaluate_probabilities(y_test, calibrated_estimator.predict_proba(x_test)[:, 1])
            result = ModelEvaluationResult(
                model_name=model_name,
                feature_set=feature_set,
                input_features=list(features),
                train_observation_count=len(x_train),
                test_observation_count=len(x_test),
                train_failure_rate=float(y_train.mean()),
                test_failure_rate=float(y_test.mean()),
                uncalibrated=uncalibrated,
                calibrated=calibrated,
                calibration_method=f"sigmoid calibration with {calibration_folds}-fold cross-validation",
                note=(
                    "Metrics are calculated on a stratified test set untouched by estimator fitting "
                    "and probability calibration. Accuracy is intentionally not reported."
                ),
            )
            artifact_name = model_artifact_name(model_name, feature_set)
            results.append(result)
            fitted_models[artifact_name] = FittedRiskModel(
                model_name=model_name,
                feature_set=feature_set,
                input_features=features,
                estimator=calibrated_estimator,
            )
    return (
        TrainingRunResult(
            target=MACHINE_FAILURE,
            random_state=random_state,
            test_fraction=test_fraction,
            results=results,
            leakage_excluded_columns=list(LEAKAGE_EXCLUDED_COLUMNS),
            note=(
                "The raw and engineering-augmented feature sets exclude the target and every "
                "failure-mode flag. Calibrated probabilities are risk evidence, not certainty."
            ),
        ),
        fitted_models,
    )


def build_pipeline(
    model_name: ModelName, features: tuple[str, ...], random_state: int
) -> Pipeline:
    """Build one reproducible preprocessing-plus-classifier pipeline."""

    categorical_features = [PRODUCT_TYPE]
    numeric_features = [feature for feature in features if feature not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("product_type", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    classifier = (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        if model_name == "logistic_regression"
        else RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    )
    return Pipeline([( "preprocessor", preprocessor), ("classifier", classifier)])


def train_and_save_models(
    frame: pd.DataFrame,
    output_directory: Path | str,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> TrainingRunResult:
    """Fit required models and persist calibrated artifacts plus transparent JSON metrics."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    result, fitted_models = train_all_models(frame, test_fraction=test_fraction, random_state=random_state)
    for artifact_name, fitted_model in fitted_models.items():
        joblib.dump(fitted_model, output_path / f"{artifact_name}.joblib")
    (output_path / "training_metrics.json").write_text(
        json.dumps(result.model_dump(), indent=2), encoding="utf-8"
    )
    return result


def features_for_set(feature_set: FeatureSetName) -> tuple[str, ...]:
    """Return the explicit allowed features for a named model feature set."""

    return RAW_FEATURES if feature_set == "raw" else ENGINEERING_AUGMENTED_FEATURES


def model_artifact_name(model_name: ModelName, feature_set: FeatureSetName) -> str:
    """Return a stable local artifact filename stem."""

    return f"{model_name}__{feature_set}"


def _assert_no_leakage(features: tuple[str, ...]) -> None:
    leaked = set(features) & set(LEAKAGE_EXCLUDED_COLUMNS)
    if leaked:
        raise ValueError(f"Target leakage in model features: {sorted(leaked)}")
