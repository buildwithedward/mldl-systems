"""Application configuration loaded from .env via pydantic's BaseSettings.
Why Pydantic BaseSettings?
- Reads from environment variables AND .env files automatically.
- Type-validates every value at startup — no silent bad config.
- One place to see all config that the app depends on"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NeuralCorp"
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 10
    log_level: str = "INFO"

    # Tell pydantic to read from .env file and environment variables
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Module level singleton instance of Settings
settings = Settings()
