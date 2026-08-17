"""Bindings from validated tool arguments to existing deterministic analytical functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.analytics.comparisons import (
    compare_failed_vs_healthy,
    compare_product_types,
)
from industrial_copilot.analytics.descriptive import (
    failure_rate_by_range,
    get_dataset_summary,
    get_failure_rate,
    get_observation,
)
from industrial_copilot.analytics.failure_analysis import failure_mode_breakdown
from industrial_copilot.analytics.models import AnalysisFilters, NumericRange
from industrial_copilot.analytics.similarity import find_similar_conditions
from industrial_copilot.data.audit import run_data_contract_audit
from industrial_copilot.features.engineering import calculate_engineering_features
from industrial_copilot.ml.predict import get_model_risk, load_fitted_risk_model
from industrial_copilot.ml.schemas import FeatureSetName, ModelName, PredictionInput
from industrial_copilot.ml.train import model_artifact_name
from industrial_copilot.statistics.confounding import analyze_conditional_relationship
from industrial_copilot.statistics.risk_regimes import discover_high_risk_regimes
from industrial_copilot.tools.registry import RegisteredTool, ToolRegistry


class ToolArguments(BaseModel):
    """Base model that rejects unrecognized tool arguments by default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FiltersArgs(ToolArguments):
    filters: AnalysisFilters = Field(default_factory=AnalysisFilters)


class FailedHealthyArgs(FiltersArgs):
    variables: list[str] = Field(min_length=1)


class RangeArgs(FiltersArgs):
    variable: str
    ranges: list[NumericRange] = Field(min_length=1)


class UidArgs(FiltersArgs):
    uid: int = Field(ge=1)


class SimilarityArgs(UidArgs):
    k: int = Field(default=5, ge=1, le=50)


class ConditionalArgs(FiltersArgs):
    exposure: str
    exposure_threshold: float
    conditioning_variable: str


class ModelRiskArgs(ToolArguments):
    model_name: ModelName = "random_forest"
    feature_set: FeatureSetName = "engineering_augmented"
    observation: PredictionInput


class EngineeringFeaturesArgs(UidArgs):
    pass


class ToolExecutionContext:
    """Application-owned data/model locations injected into fixed tool closures."""

    def __init__(self, frame: pd.DataFrame, models_directory: Path) -> None:
        self.frame = frame
        self.models_directory = models_directory


def build_industrial_registry(context: ToolExecutionContext) -> ToolRegistry:
    """Register each evidence tool once and delegate to validated domain logic."""

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            "get_dataset_summary",
            "Summarize observations, failure prevalence, product mix, and operating variables.",
            FiltersArgs,
            lambda args: get_dataset_summary(context.frame, args.filters).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "get_failure_rate",
            "Calculate selected machine-failure count and rate.",
            FiltersArgs,
            lambda args: get_failure_rate(context.frame, args.filters).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "compare_failed_vs_healthy",
            "Compare selected numeric operating variables for failed and healthy observations.",
            FailedHealthyArgs,
            lambda args: compare_failed_vs_healthy(
                context.frame, args.variables, args.filters
            ).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "failure_rate_by_range",
            "Calculate failure rate in explicit caller-planned numeric ranges.",
            RangeArgs,
            lambda args: failure_rate_by_range(
                context.frame, args.variable, args.ranges, args.filters
            ).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "failure_mode_breakdown",
            "Break down the selected observations by potentially overlapping failure-mode flags.",
            FiltersArgs,
            lambda args: failure_mode_breakdown(context.frame, args.filters).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "compare_product_types",
            "Compare L, M, and H product types by selected failure prevalence.",
            FiltersArgs,
            lambda args: compare_product_types(context.frame, args.filters).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "get_observation",
            "Retrieve one UID-keyed observation with calculated engineering features.",
            UidArgs,
            lambda args: get_observation(context.frame, args.uid).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "calculate_engineering_features",
            "Return calculated engineering fields for one UID without changing raw data.",
            EngineeringFeaturesArgs,
            lambda args: _engineering_fields_for_uid(context.frame, args.uid),
        )
    )
    registry.register(
        RegisteredTool(
            "find_similar_conditions",
            "Retrieve standardized nearest historical operating conditions for one UID.",
            SimilarityArgs,
            lambda args: find_similar_conditions(context.frame, args.uid, args.k, args.filters).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "run_data_contract_audit",
            "Audit source schema, labels, ranges, duplicates, and documented-rule consistency.",
            ToolArguments,
            lambda _args: run_data_contract_audit(context.frame).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "analyze_conditional_relationship",
            "Quantify an exposure before and after stratification/continuous adjustment.",
            ConditionalArgs,
            lambda args: analyze_conditional_relationship(
                context.frame,
                args.exposure,
                args.exposure_threshold,
                args.conditioning_variable,
                args.filters,
            ).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "discover_high_risk_regimes",
            "Discover and internally confirm interpretable two-condition high-risk regimes.",
            FiltersArgs,
            lambda args: discover_high_risk_regimes(context.frame, args.filters).model_dump(),
        )
    )
    registry.register(
        RegisteredTool(
            "get_model_risk",
            "Return a calibrated local-model risk estimate from telemetry with no label inputs.",
            ModelRiskArgs,
            lambda args: _model_risk(context.models_directory, args),
        )
    )
    return registry


def _engineering_fields_for_uid(frame: pd.DataFrame, uid: int) -> dict[str, Any]:
    engineered = calculate_engineering_features(frame)
    matched = engineered.loc[engineered["UID"].eq(uid)]
    if matched.empty:
        raise KeyError(f"No observation found for UID {uid}.")
    row = matched.iloc[0]
    selected_columns = [
        "Temperature delta [K]",
        "Angular velocity [rad/s]",
        "Mechanical power [W]",
        "Overstrain load [min Nm]",
        "Overstrain threshold [min Nm]",
        "HDF documented rule condition",
        "PWF documented rule condition",
        "OSF documented rule condition",
    ]
    return {column: _json_value(row[column]) for column in selected_columns}


def _model_risk(models_directory: Path, arguments: ModelRiskArgs) -> dict[str, Any]:
    artifact = models_directory / f"{model_artifact_name(arguments.model_name, arguments.feature_set)}.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Local model artifact is missing: {artifact}. Run scripts/train_models.py.")
    return get_model_risk(load_fitted_risk_model(artifact), arguments.observation).model_dump()


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
