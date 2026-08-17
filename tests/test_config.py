from industrial_copilot.config import PROJECT_ROOT, Settings


def test_settings_have_safe_offline_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.project_name == "Industrial Intelligence Copilot"
    assert settings.llm_enabled is False
    assert settings.raw_data_path == PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
