# All config comes from environment variables, with defaults provided where appropriate.
# This allows for easy configuration in different environments (local, staging, production) without code changes
# Pydantic settings reads from .env file in development, and from actual environment variables in production

import logging
from pydantic_settings import BaseSettings # reads from environment variables and .env files automatically
from pydantic import Field # add validation rules and default values to settings fields

# Configuring logging once at module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """NeuralCorp application settings loaded from environment variables.

    All fields map 1:1 to environment variable names (uppercase).
    Pydantic validates types and raises ConfigurationError on startup
    if any required variable is missing — fail fast, fail loudly.
    """
    app_name: str=Field("NeuralCorp ML Template", description="The name of the application")
    environment: str=Field("development", description="The application environment (development, staging, production)")
    log_level: str=Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    data_dir: str=Field("data", description="Directory where data files are stored")
    model_dir: str=Field("models", description="Directory where model files are stored")
    random_seed: int=Field(42, description="Random seed for reproducibility")

    class Config:
        env_file = ".env" # Automatically read from .env file in development
        env_file_encoding = 'utf-8' # Ensure proper encoding when reading .env files
        case_sensitive = False # Environment variable names are case-insensitive

# Create a global settings instance that can be imported and used throughout the application
def get_settings() -> Settings:
    """Load settings from environment variables and return a Settings instance."""
    settings = Settings()
    logger.info("Settings Loaded",
                extra={
                    "app_name": settings.app_name,
                    "environment": settings.environment,
                    "log_level": settings.log_level
                    }
                )
    return settings
