"""Configuration management using Pydantic."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration from .env."""

    log_level: str = "INFO"
    log_file: str = "output/logs/pipeline.json.log"
    plot_output_dir: str = "output/plots"
    data_seed: int = 42
    num_students: int = 500

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def plot_dir_path(self) -> Path:
        """Return plot directory as Path object."""
        return Path(self.plot_output_dir)

    @property
    def log_file_path(self) -> Path:
        """Return log file as Path object."""
        return Path(self.log_file)


settings = Settings()