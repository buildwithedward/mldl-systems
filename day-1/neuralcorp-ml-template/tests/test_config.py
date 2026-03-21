from unittest.mock import patch # mock env vars for testing
from src.config import get_settings, Settings


class TestSettings:
    "Unit tests for the Settings class and get_settings function."


    def test_default_settings(self):
        "Settings should load default values when no environment variables are set."
        settings = get_settings()
        assert settings.app_name == "NeuralCorp-ML-Template".lower() # case-insensitive
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.data_dir == "data"
        assert settings.model_dir == "models"
        assert settings.random_seed == 42


    def test_env_var_overrides(self):
        "Settings should override defaults with environment variables."
        with patch.dict('os.environ', {
            'APP_NAME': 'Test App',
            'ENVIRONMENT': 'production',
            'LOG_LEVEL': 'DEBUG',
            'DATA_DIR': '/tmp/data',
            'MODEL_DIR': '/tmp/models',
            'RANDOM_SEED': '123'
        }):
            settings = get_settings()
            assert settings.app_name == "Test App"
            assert settings.environment == "production"
            assert settings.log_level == "DEBUG"
            assert settings.data_dir == "/tmp/data"
            assert settings.model_dir == "/tmp/models"
            assert settings.random_seed == 123


    def test_random_seed_is_int(self):
        "Random seed must be an integer; non-integer values should raise a validation error."
        settings = get_settings()
        assert isinstance(settings.random_seed, int)


    def test_settings_returns_settings_instance(self):
        "get_settings should return an instance of the Settings class."
        settings = get_settings()
        assert isinstance(settings, Settings)