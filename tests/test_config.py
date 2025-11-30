"""Tests for configuration management"""
import os
import tempfile
import pytest
from ai_drawing_analyzer.utils.config import AppConfig, load_env_file


def test_app_config_defaults():
    """Test that default configuration values are correct"""
    assert AppConfig.PDF_ZOOM_LEVEL == 2
    assert AppConfig.JPEG_QUALITY == 90
    assert AppConfig.API_TIMEOUT == 120.0
    assert AppConfig.MAX_RETRIES == 3


def test_app_config_from_env():
    """Test loading configuration from environment variables"""
    os.environ["PDF_ZOOM_LEVEL"] = "3"
    os.environ["JPEG_QUALITY"] = "85"

    config = AppConfig.from_env()

    assert config["pdf_zoom_level"] == 3
    assert config["jpeg_quality"] == 85

    # Cleanup
    del os.environ["PDF_ZOOM_LEVEL"]
    del os.environ["JPEG_QUALITY"]


def test_load_env_file():
    """Test loading environment variables from .env file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("TEST_KEY=test_value\n")
        f.write("# This is a comment\n")
        f.write('QUOTED_VALUE="quoted"\n')
        f.write("\n")  # Empty line
        env_file = f.name

    try:
        env_vars = load_env_file(env_file)

        assert "TEST_KEY" in env_vars
        assert env_vars["TEST_KEY"] == "test_value"
        assert "QUOTED_VALUE" in env_vars
        assert env_vars["QUOTED_VALUE"] == "quoted"
    finally:
        os.unlink(env_file)


def test_load_env_file_nonexistent():
    """Test that loading a non-existent .env file returns empty dict"""
    env_vars = load_env_file("nonexistent.env")
    assert env_vars == {}
