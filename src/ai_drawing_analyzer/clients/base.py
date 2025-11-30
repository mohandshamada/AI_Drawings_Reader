from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class BaseClient(ABC):
    """
    Abstract base class for AI vision API clients.

    All concrete clients must implement methods for listing available models
    and analyzing images with vision-language models.

    Attributes:
        api_key: API authentication key (None for local inference clients)
        timeout: Request timeout in seconds (default: 120.0)

    Example:
        >>> from ai_drawing_analyzer.clients.gemini import GeminiClient
        >>> client = GeminiClient(api_key="your_key")
        >>> models = await client.get_available_models()
        >>> result = await client.analyze_image(img_b64, "Extract text", "gemini-2.0-flash-exp")
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client.

        Args:
            api_key: Optional API key for authentication
        """
        self.api_key = api_key
        self.timeout = 120.0

    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get list of available models for this client.

        Returns:
            List of dicts with 'id' and 'name' keys for each model

        Example:
            [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"}
            ]
        """
        pass

    @abstractmethod
    async def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        """
        Analyze an image using a vision-language model.

        Args:
            image_base64: Base64 encoded image data
            prompt: Text prompt for analysis
            model: Model identifier to use

        Returns:
            Text response from the model

        Raises:
            ValueError: If required parameters are missing
            httpx.HTTPError: If API request fails
        """
        pass

    def _get_retry_decorator(self):
        """
        Get tenacity retry decorator for network operations.

        Returns:
            Configured retry decorator with exponential backoff
        """
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout))
        )
