"""Application configuration loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings with safe offline defaults for deterministic operation."""

    project_name: str = "Industrial Intelligence Copilot"
    llm_enabled: bool = False
    llm_provider: Literal["gemini", "groq"] = "groq"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 6.0
    agentic_planner_enabled: bool = True
    agent_deep_mode_enabled: bool = True
    agent_max_tool_rounds: int = 2
    agent_max_initial_tools: int = 4
    agent_max_review_tools: int = 2
    knowledge_enabled: bool = True
    knowledge_top_k: int = 3
    grounded_numbers_enabled: bool = True
    agent_timeout_seconds: float = 20.0
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
    processed_data_path: Path = PROJECT_ROOT / "data" / "processed" / "ai4i2020_features.parquet"
    live_warning_risk_threshold: float = 0.20
    live_incident_risk_threshold: float = 0.35
    live_osf_warning_margin_min_nm: float = 1000.0
    live_hdf_temperature_margin_k: float = 0.5
    live_hdf_rpm_margin: float = 50.0
    live_pwf_power_margin_w: float = 500.0

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached, validated settings object for the current process."""

    return Settings()
