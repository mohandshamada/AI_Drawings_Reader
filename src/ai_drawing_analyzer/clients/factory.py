import os
from typing import Optional
from .base import BaseClient
from .gemini import GeminiClient
from .huggingface import HuggingFaceRouterClient, HuggingFaceLocalClient
from .openai import OpenAIClient, OpenRouterClient
from .anthropic import AnthropicClient

class ClientFactory:
    PROVIDERS = {
        'huggingface-local': HuggingFaceLocalClient,
        'huggingface': HuggingFaceRouterClient,
        'openrouter': OpenRouterClient,
        'gemini': GeminiClient,
        'openai': OpenAIClient,
        'anthropic': AnthropicClient
    }

    API_KEY_MAP = {
        'huggingface-local': None,
        'huggingface': 'HF_TOKEN',
        'openrouter': 'OPENROUTER_API_KEY',
        'gemini': 'GOOGLE_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY'
    }

    @staticmethod
    def get_api_key(provider: str, override_key: Optional[str] = None) -> Optional[str]:
        if override_key: return override_key
        if provider == 'huggingface-local': return None
        
        key_name = ClientFactory.API_KEY_MAP.get(provider)
        if key_name:
            return os.environ.get(key_name)
        return None

    @staticmethod
    def create_client(provider: str, api_key: Optional[str] = None, model_id: Optional[str] = None) -> BaseClient:
        provider = provider.lower()
        if provider not in ClientFactory.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        
        api_key = ClientFactory.get_api_key(provider, api_key)
        client_class = ClientFactory.PROVIDERS[provider]
        
        if provider == 'huggingface-local':
            return client_class(model_id=model_id)
        else:
            if not api_key:
                 # Try to fallback to alternative environment variable names if standard ones fail
                if provider == 'huggingface' and os.environ.get('HUGGINGFACE_API_KEY'):
                    api_key = os.environ.get('HUGGINGFACE_API_KEY')
                else:
                    raise ValueError(f"API Key required for {provider}. Set {ClientFactory.API_KEY_MAP.get(provider)} in .env")
            return client_class(api_key)
