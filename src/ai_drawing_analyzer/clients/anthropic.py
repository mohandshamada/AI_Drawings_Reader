from typing import List, Dict
import httpx
from .base import BaseClient

class AnthropicClient(BaseClient):
    """Anthropic Claude API client"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.anthropic.com/v1"
    
    async def get_available_models(self) -> List[Dict[str, str]]:
        return [{"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"}, {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"}]
    
    async def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        payload = {
            "model": model, "max_tokens": 2048,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                         {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}}]}]
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/messages", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['content'][0]['text']
