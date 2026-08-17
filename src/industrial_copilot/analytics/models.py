"""Small, serializable result models returned by deterministic analytics."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from industrial_copilot.data.schema import FAILURE_MODES, VALID_PRODUCT_TYPES


class NumericRange(BaseModel):
    """Inclusive numeric filter or requested numeric analysis range."""

    model_config = ConfigDict(frozen=True)

    column: str
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> NumericRange:
        if self.minimum is None and self.maximum is None:
            raise ValueError("A numeric range needs a minimum, maximum, or both.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum.")
        return self


class AnalysisFilters(BaseModel):
    """Explicit subset rules used by every analytical function."""

    model_config = ConfigDict(frozen=True)

    product_types: list[str] | None = None
    machine_failure: int | None = Field(default=None, ge=0, le=1)
    failure_mode: str | None = None
    numeric_ranges: list[NumericRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_categories(self) -> AnalysisFilters:
        if self.product_types is not None:
            invalid = set(self.product_types) - VALID_PRODUCT_TYPES
            if invalid:
                raise ValueError(f"Unknown product type(s): {sorted(invalid)}")
        if self.failure_mode is not None and self.failure_mode not in FAILURE_MODES:
            raise ValueError(f"Unknown failure mode: {self.failure_mode}")
        return self


class SummaryStatistics(BaseModel):
    """Basic descriptive statistics for one numeric variable."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    standard_deviation: float | None = None


class DatasetSummary(BaseModel):
    """Dataset size, failure prevalence, category counts, and numeric summaries."""

    model_config = ConfigDict(frozen=True)

    observation_count: int = Field(ge=0)
    failed_observation_count: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    product_type_counts: dict[str, int]
    numeric_summary: dict[str, SummaryStatistics]
    filters: AnalysisFilters


class FailureRateResult(BaseModel):
    """Failure rate for a selected subset."""

    model_config = ConfigDict(frozen=True)

    observation_count: int = Field(ge=0)
    failed_observation_count: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    filters: AnalysisFilters


class RangeFailureRate(BaseModel):
    """Failure rate within one non-overlapping numeric interval."""

    model_config = ConfigDict(frozen=True)

    label: str
    minimum: float | None = None
    maximum: float | None = None
    observation_count: int = Field(ge=0)
    failed_observation_count: int = Field(ge=0)
    failure_rate: float | None = Field(default=None, ge=0, le=1)


class RangeAnalysis(BaseModel):
    """Failure-rate breakdown for explicit numeric ranges."""

    model_config = ConfigDict(frozen=True)

    variable: str
    ranges: list[RangeFailureRate]
    filters: AnalysisFilters


class FailedHealthyComparison(BaseModel):
    """Side-by-side descriptive statistics for failed and healthy observations."""

    model_config = ConfigDict(frozen=True)

    variables: dict[str, dict[str, SummaryStatistics]]
    filters: AnalysisFilters


class FailureModeBreakdownItem(BaseModel):
    """Prevalence of one failure-mode flag; modes may overlap."""

    model_config = ConfigDict(frozen=True)

    failure_mode: str
    flagged_observation_count: int = Field(ge=0)
    rate_among_selected: float = Field(ge=0, le=1)
    share_of_failed_observations: float | None = Field(default=None, ge=0)


class FailureModeBreakdown(BaseModel):
    """Mode-level evidence, retaining potential overlap among labels."""

    model_config = ConfigDict(frozen=True)

    selected_observation_count: int = Field(ge=0)
    selected_failed_observation_count: int = Field(ge=0)
    modes: list[FailureModeBreakdownItem]
    filters: AnalysisFilters
    note: str


class ProductTypeFailureRate(BaseModel):
    """Failure prevalence for one L/M/H product group."""

    model_config = ConfigDict(frozen=True)

    product_type: str
    observation_count: int = Field(ge=0)
    failed_observation_count: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)


class ProductTypeComparison(BaseModel):
    """Failure-rate comparison across product types after any other filters."""

    model_config = ConfigDict(frozen=True)

    groups: list[ProductTypeFailureRate]
    filters: AnalysisFilters


class ObservationRecord(BaseModel):
    """One engineered observation, retained for later alert investigation."""

    model_config = ConfigDict(frozen=True)

    uid: int
    values: dict[str, bool | float | int | str | None]


class FeatureDifference(BaseModel):
    """The difference between a target observation and one similar observation."""

    model_config = ConfigDict(frozen=True)

    feature: str
    target_value: float
    similar_value: float
    absolute_standardized_difference: float


class SimilarObservation(BaseModel):
    """Nearest historical operating condition with interpretable differences."""

    model_config = ConfigDict(frozen=True)

    uid: int
    distance: float = Field(ge=0)
    machine_failure: int = Field(ge=0, le=1)
    active_failure_modes: list[str]
    important_differences: list[FeatureDifference]


class SimilarConditionsResult(BaseModel):
    """Nearest-neighbour evidence from standardized engineering variables."""

    model_config = ConfigDict(frozen=True)

    target_uid: int
    candidate_count: int = Field(ge=0)
    similar_case_failure_rate: float | None = Field(default=None, ge=0, le=1)
    feature_columns: list[str]
    observations: list[SimilarObservation]
    filters: AnalysisFilters
