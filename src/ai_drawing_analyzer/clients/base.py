from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class BaseClient(ABC):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.timeout = 120.0

    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, str]]:
        pass

    @abstractmethod
    async def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        pass
    
    def _get_retry_decorator(self):
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout))
        )
