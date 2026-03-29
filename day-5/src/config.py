"""Configuration management using Pydantic BaseSettings."""

import logging
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration loaded from .env and environment variables."""

    log_level: str = "INFO"
    data_dir: str = "./data"
    max_file_size_mb: int = 100
    analysis_enabled: bool = True

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = False

    def get_data_path(self) -> Path:
        """Return Path object for data directory.

        Returns:
            Path: Data directory as pathlib.Path object.
        """
        path = Path(self.data_dir)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {path}")
        return path


def load_settings() -> Settings:
    """Load and validate settings.

    Returns:
        Settings: Loaded configuration object.

    Raises:
        ConfigError: If settings are invalid.
    """
    try:
        settings = Settings()
        logger.info(f"Settings loaded: log_level={settings.log_level}")
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise