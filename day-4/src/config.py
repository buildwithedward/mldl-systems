"""Pydantic based settings for configuration management."""

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    app_name: str = "nc-004-python-foundations"
    app_env: str = "development"
    log_level: str = "INFO"
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10


# Singleton — import this object everywhere instead of re-reading .env
settings = Settings()


# Configure root logger based on settings
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)