from typing import List, Dict
import httpx
from .base import BaseClient

class OpenAIClient(BaseClient):
    """OpenAI API client"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"

    async def get_available_models(self) -> List[Dict[str, str]]:
        return [{"id": "gpt-4o", "name": "GPT-4o"}, {"id": "gpt-4o-mini", "name": "GPT-4o Mini"}]

    async def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        if not image_base64 or not prompt or not model:
            raise ValueError("image_base64, prompt, and model are required")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            "max_tokens": 2048
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Validate response structure
            if not data.get('choices') or len(data['choices']) == 0:
                raise ValueError(f"Invalid response structure: {data}")

            content = data['choices'][0].get('message', {}).get('content')
            if not content:
                raise ValueError("Empty content in response")

            return content

class OpenRouterClient(BaseClient):
    """OpenRouter API client for accessing multiple models"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://openrouter.ai/api/v1"

    async def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
            {"id": "openai/gpt-4o", "name": "GPT-4o"},
            {"id": "google/gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash"},
            {"id": "meta-llama/llama-3.2-90b-vision-instruct", "name": "Llama 3.2 90B Vision"},
        ]

    async def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        if not image_base64 or not prompt or not model:
            raise ValueError("image_base64, prompt, and model are required")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mohandshamada/AI_Drawings_Reader",
            "X-Title": "AI Drawing Analyzer"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            "max_tokens": 2048
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Validate response structure
            if not data.get('choices') or len(data['choices']) == 0:
                raise ValueError(f"Invalid response structure: {data}")

            content = data['choices'][0].get('message', {}).get('content')
            if not content:
                raise ValueError("Empty content in response")

            return content
