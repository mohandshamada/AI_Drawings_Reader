import os
from typing import Optional
from .base import BaseClient
from .gemini import GeminiClient
from .huggingface import HuggingFaceRouterClient, HuggingFaceLocalClient
from .openai import OpenAIClient, OpenRouterClient
from .anthropic import AnthropicClient


class ClientFactory:
    """
    Factory for creating AI vision client instances.

    Supports multiple providers including local inference and cloud APIs.
    Automatically handles API key retrieval from environment variables.

    Providers:
        - huggingface-local: Local inference with transformers
        - huggingface: HuggingFace Router API
        - openrouter: OpenRouter API (multi-model)
        - gemini: Google Gemini API
        - openai: OpenAI API
        - anthropic: Anthropic Claude API

    Example:
        >>> from ai_drawing_analyzer.clients.factory import ClientFactory
        >>> client = ClientFactory.create_client('gemini', api_key='your_key')
        >>> # Or use environment variable
        >>> import os
        >>> os.environ['GOOGLE_API_KEY'] = 'your_key'
        >>> client = ClientFactory.create_client('gemini')
    """

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
        """
        Get API key for a provider from override or environment.

        Args:
            provider: Provider name
            override_key: Optional API key to use instead of environment

        Returns:
            API key string or None for local providers
        """
        if override_key:
            return override_key
        if provider == 'huggingface-local':
            return None

        key_name = ClientFactory.API_KEY_MAP.get(provider)
        if key_name:
            return os.environ.get(key_name)
        return None

    @staticmethod
    def create_client(provider: str, api_key: Optional[str] = None, model_id: Optional[str] = None) -> BaseClient:
        """
        Create a client instance for the specified provider.

        Args:
            provider: Provider name (e.g., 'gemini', 'openai', 'huggingface-local')
            api_key: Optional API key (will use environment variable if not provided)
            model_id: Model ID for local inference providers

        Returns:
            Configured client instance

        Raises:
            ValueError: If provider is unknown or API key is missing

        Example:
            >>> client = ClientFactory.create_client('gemini', api_key='your_key')
            >>> local_client = ClientFactory.create_client(
            ...     'huggingface-local',
            ...     model_id='microsoft/Florence-2-large'
            ... )
        """
        provider = provider.lower()
        if provider not in ClientFactory.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        api_key = ClientFactory.get_api_key(provider, api_key)
        client_class = ClientFactory.PROVIDERS[provider]

        if provider == 'huggingface-local':
            return client_class(model_id=model_id)
        else:
            if not api_key:
                # Try to fallback to alternative environment variable names
                if provider == 'huggingface' and os.environ.get('HUGGINGFACE_API_KEY'):
                    api_key = os.environ.get('HUGGINGFACE_API_KEY')
                else:
                    raise ValueError(
                        f"API Key required for {provider}. "
                        f"Set {ClientFactory.API_KEY_MAP.get(provider)} in .env or use --api-key"
                    )
            return client_class(api_key)
