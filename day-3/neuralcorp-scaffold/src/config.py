"""Configuration loaded from environment variables"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic BaseSettings automatically reads from environment variables
    and .env files. This is the production-standard way to handle config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Project Metadata
    project_name: str = "NeuralCorp"
    environment: str = "dev"
    log_level: str = "INFO"

    # model config
    model_path: str = "models/model.pkl"
    batch_size: int = 32
    random_seed: int = 42


# Single global instance — import this everywhere
settings = Settings()
