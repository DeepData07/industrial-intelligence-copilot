"""Small, serializable outputs for the statistical evidence layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.analytics.models import AnalysisFilters


class ConfidenceInterval(BaseModel):
    """Two-sided confidence interval for an effect estimate."""

    model_config = ConfigDict(frozen=True)

    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    lower: float | None = None
    upper: float | None = None


class BinaryAssociation(BaseModel):
    """Transparent 2×2 association between a binary exposure and machine failure."""

    model_config = ConfigDict(frozen=True)

    exposed_observation_count: int = Field(ge=0)
    exposed_failure_count: int = Field(ge=0)
    unexposed_observation_count: int = Field(ge=0)
    unexposed_failure_count: int = Field(ge=0)
    exposed_failure_rate: float | None = Field(default=None, ge=0, le=1)
    unexposed_failure_rate: float | None = Field(default=None, ge=0, le=1)
    risk_difference: float | None = None
    risk_difference_ci: ConfidenceInterval
    risk_ratio: float | None = Field(default=None, ge=0)
    risk_ratio_ci: ConfidenceInterval
    odds_ratio: float | None = Field(default=None, ge=0)
    odds_ratio_ci: ConfidenceInterval
    fisher_exact_p_value: float | None = Field(default=None, ge=0, le=1)
    continuity_correction_applied: bool


class StratumAssociation(BaseModel):
    """A binary association estimate within one conditioning stratum."""

    model_config = ConfigDict(frozen=True)

    stratum: str
    association: BinaryAssociation


class MantelHaenszelEstimate(BaseModel):
    """Common odds ratio across valid 2×2 strata."""

    model_config = ConfigDict(frozen=True)

    available: bool
    stratum_count: int = Field(ge=0)
    adjusted_odds_ratio: float | None = Field(default=None, ge=0)
    confidence_interval: ConfidenceInterval
    null_odds_p_value: float | None = Field(default=None, ge=0, le=1)
    homogeneity_p_value: float | None = Field(default=None, ge=0, le=1)
    note: str


class ContinuousLogisticEvidence(BaseModel):
    """Adjusted logistic association per one standard deviation of a numeric exposure."""

    model_config = ConfigDict(frozen=True)

    available: bool
    exposure_odds_ratio_per_standard_deviation: float | None = Field(default=None, ge=0)
    confidence_interval: ConfidenceInterval
    p_value: float | None = Field(default=None, ge=0, le=1)
    adjustment_variables: list[str]
    note: str


EffectChange = Literal[
    "CONFIRMED_REVERSAL",
    "RELATIONSHIP_WEAKENED",
    "NO_MEANINGFUL_CHANGE",
    "INSUFFICIENT_SUPPORT",
]


class ConditionalRelationshipResult(BaseModel):
    """Unadjusted and conditioned evidence for a user-specified exposure comparison."""

    model_config = ConfigDict(frozen=True)

    exposure: str
    exposure_threshold: float
    conditioning_variable: str
    conditioning_method: str
    selected_observation_count: int = Field(ge=0)
    aggregate_association: BinaryAssociation
    stratum_associations: list[StratumAssociation]
    mantel_haenszel: MantelHaenszelEstimate
    continuous_logistic_evidence: ContinuousLogisticEvidence
    effect_change: EffectChange
    interpretation: str
    filters: AnalysisFilters


class AdjustedPValue(BaseModel):
    """One hypothesis' Benjamini-Hochberg adjusted significance result."""

    model_config = ConfigDict(frozen=True)

    identifier: str
    p_value: float | None = Field(default=None, ge=0, le=1)
    q_value: float | None = Field(default=None, ge=0, le=1)
    rejected: bool = False


class RegimeCondition(BaseModel):
    """One readable predicate in an automatically discovered operating regime."""

    model_config = ConfigDict(frozen=True)

    feature: str
    operator: Literal[">=", "<="]
    threshold: float

    @property
    def label(self) -> str:
        return f"{self.feature} {self.operator} {self.threshold:.3g}"


class RegimePartitionEvidence(BaseModel):
    """Support and association statistics for discovery or held-out confirmation data."""

    model_config = ConfigDict(frozen=True)

    observation_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    baseline_failure_rate: float | None = Field(default=None, ge=0, le=1)
    risk_ratio: float | None = Field(default=None, ge=0)
    risk_lift: float | None = None
    odds_ratio: float | None = Field(default=None, ge=0)
    odds_ratio_ci: ConfidenceInterval
    fisher_exact_p_value: float | None = Field(default=None, ge=0, le=1)
    dominant_failure_mode: str | None = None


RegimeStatus = Literal["EXPLORATORY", "CONFIRMED", "NOT_STABLE", "INSUFFICIENT_DATA"]


class RiskRegime(BaseModel):
    """A two-condition, interpretable operating regime with separated discovery/confirmation."""

    model_config = ConfigDict(frozen=True)

    conditions: list[RegimeCondition]
    status: RegimeStatus
    discovery: RegimePartitionEvidence
    confirmation: RegimePartitionEvidence
    discovery_q_value: float | None = Field(default=None, ge=0, le=1)
    discovery_fdr_rejected: bool
    confirmation_stable: bool | None = None


class RiskRegimeDiscoveryResult(BaseModel):
    """Output of a held-out hidden-risk-regime search."""

    model_config = ConfigDict(frozen=True)

    discovery_observation_count: int = Field(ge=0)
    confirmation_observation_count: int = Field(ge=0)
    tested_regime_count: int = Field(ge=0)
    minimum_discovery_support: int = Field(ge=1)
    minimum_confirmation_support: int = Field(ge=1)
    fdr_alpha: float = Field(gt=0, lt=1)
    regimes: list[RiskRegime]
    filters: AnalysisFilters
    note: str
