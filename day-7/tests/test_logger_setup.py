"""Tests for logging configuration."""
import pytest
import logging
import json
from pathlib import Path
from src.logger_setup import setup_logging, get_logger
from src.config import Settings
from src.exceptions import LoggingConfigError


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_returns_logger(self, tmp_path: Path):
        """Test that setup_logging returns a Logger instance."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_creates_log_file(self, tmp_path: Path, monkeypatch):
        """Test that log file is created."""
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("LOG_FILE", str(log_file))

        logger = setup_logging()
        logger.info("Test message")

        assert log_file.exists()

    def test_json_format_is_valid(self, tmp_path: Path, monkeypatch):
        """Test that JSON logs are valid JSON."""
        log_file = tmp_path / "test.json.log"
        monkeypatch.setenv("LOG_FILE", str(log_file))

        logger = setup_logging()
        logger.info("Test message", extra={"key": "value"})

        with open(log_file) as f:
            line = f.readline()
            log_entry = json.loads(line)
            assert log_entry["message"] == "Test message"
            assert log_entry["key"] == "value"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_with_correct_name(self):
        """Test that get_logger returns a Logger with correct name."""
        logger = get_logger("test_module")
        assert logger.name == "test_module"

    def test_returns_same_logger_on_repeated_calls(self):
        """Test that repeated calls return the same logger."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        assert logger1 is logger2