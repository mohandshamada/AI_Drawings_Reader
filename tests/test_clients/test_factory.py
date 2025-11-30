"""Tests for client factory"""
import os
import pytest
from ai_drawing_analyzer.clients.factory import ClientFactory
from ai_drawing_analyzer.clients.gemini import GeminiClient
from ai_drawing_analyzer.clients.openai import OpenAIClient


def test_client_factory_providers():
    """Test that all expected providers are registered"""
    expected_providers = [
        'huggingface-local',
        'huggingface',
        'openrouter',
        'gemini',
        'openai',
        'anthropic'
    ]

    for provider in expected_providers:
        assert provider in ClientFactory.PROVIDERS


def test_client_factory_api_key_map():
    """Test that API key mapping is correct"""
    assert ClientFactory.API_KEY_MAP['gemini'] == 'GOOGLE_API_KEY'
    assert ClientFactory.API_KEY_MAP['openai'] == 'OPENAI_API_KEY'
    assert ClientFactory.API_KEY_MAP['anthropic'] == 'ANTHROPIC_API_KEY'
    assert ClientFactory.API_KEY_MAP['huggingface-local'] is None


def test_create_client_with_api_key():
    """Test creating a client with direct API key"""
    client = ClientFactory.create_client('gemini', api_key='test_key')
    assert isinstance(client, GeminiClient)
    assert client.api_key == 'test_key'


def test_create_client_missing_api_key():
    """Test that creating a client without API key raises error"""
    # Make sure the env var is not set
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']

    with pytest.raises(ValueError, match="API Key required"):
        ClientFactory.create_client('openai')


def test_create_client_from_env():
    """Test creating a client with API key from environment"""
    os.environ['GOOGLE_API_KEY'] = 'env_test_key'

    try:
        client = ClientFactory.create_client('gemini')
        assert isinstance(client, GeminiClient)
        assert client.api_key == 'env_test_key'
    finally:
        del os.environ['GOOGLE_API_KEY']


def test_create_client_unknown_provider():
    """Test that creating client with unknown provider raises error"""
    with pytest.raises(ValueError, match="Unknown provider"):
        ClientFactory.create_client('unknown_provider')
