"""HTTP request and response contracts; analytical contracts remain in their own modules."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.analytics.models import AnalysisFilters
from industrial_copilot.copilot.state import ConversationState
from industrial_copilot.ml.schemas import FeatureSetName, ModelName, PredictionInput


class HealthResponse(BaseModel):
    """Safe operational status without exposing API keys or other secrets."""

    model_config = ConfigDict(frozen=True)

    status: str
    project_name: str
    llm_enabled: bool
    llm_provider: str


class QuestionRequest(BaseModel):
    """One natural-language request plus compact public conversation context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1, max_length=1_000)
    state: ConversationState | None = None


class SimilarRequest(BaseModel):
    """Request nearest AI4I conditions for one observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uid: int = Field(ge=1)
    k: int = Field(default=5, ge=1, le=50)
    filters: AnalysisFilters = Field(default_factory=AnalysisFilters)


class PredictionRequest(BaseModel):
    """Leakage-safe prediction request containing telemetry only, never outcome labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: ModelName = "random_forest"
    feature_set: FeatureSetName = "engineering_augmented"
    observation: PredictionInput


class AlertRequest(BaseModel):
    """Replay one AI4I observation as clearly labelled simulated alert telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uid: int = Field(ge=1)
    k: int = Field(default=5, ge=1, le=50)
    model_name: ModelName = "random_forest"
    feature_set: FeatureSetName = "engineering_augmented"


class AlertResponse(BaseModel):
    """Evidence required by the later Explain This Alert experience."""

    model_config = ConfigDict(frozen=True)

    telemetry_source: str = "SIMULATED AI4I TELEMETRY"
    observation: dict[str, Any]
    engineering_features: dict[str, Any]
    similar_conditions: dict[str, Any]
    model_risk: dict[str, Any]


class LiveStateRequest(BaseModel):
    """One deterministic snapshot request for the disclosed live demo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: str = Field(default="OSF", pattern="^(OSF|HDF|PWF|REPLAY)$")
    cycle: int = Field(default=0, ge=0, le=10_000)


class LiveConversationTurn(BaseModel):
    """One bounded public conversation turn supplied for follow-up context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class LiveCopilotRequest(LiveStateRequest):
    """Evidence-backed question tied to a deterministic live snapshot."""

    question: str = Field(min_length=1, max_length=1_000)
    conversation: list[LiveConversationTurn] = Field(default_factory=list, max_length=6)
    mode: Literal["quick", "deep"] = "quick"


class LiveWhatIfRequest(LiveStateRequest):
    """Proposed telemetry point evaluated through the same real twin pipeline."""

    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)
