"""Configuration for the data pipeline."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # Paths
    data_dir: str = "data"
    log_dir: str = "logs"

    # Data generation
    num_students: int = 100
    num_subjects: int = 5
    num_exams: int = 4
    seed: int = 42

    # Logging
    log_level: str = "INFO"
    log_file: str = "pipeline.log"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Ensure directories exist
Path(settings.data_dir).mkdir(exist_ok=True)
Path(settings.log_dir).mkdir(exist_ok=True)