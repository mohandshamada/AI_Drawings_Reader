from typing import List, Dict
import httpx
import asyncio
from .base import BaseClient

class GeminiClient(BaseClient):
    """Google Gemini API client"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash (Experimental)"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
        ]
    
    async def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        if not image_base64 or not prompt or not model:
            raise ValueError("image_base64, prompt, and model are required")

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }

        # Manual retry logic for 429 as per original code, but using async sleep
        max_retries = 3
        retry_count = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while retry_count <= max_retries:
                try:
                    response = await client.post(url, json=payload)

                    if response.status_code == 429:
                        await asyncio.sleep(65)
                        retry_count += 1
                        continue

                    response.raise_for_status()
                    result = response.json()

                    if 'promptFeedback' in result and result['promptFeedback'].get('blockReason'):
                        raise Exception(f"Blocked by safety settings: {result['promptFeedback']}")

                    if not result.get('candidates') or len(result['candidates']) == 0:
                        raise ValueError(f"Invalid response structure: {result}")

                    content = result['candidates'][0].get('content', {}).get('parts', [])
                    if not content or not content[0].get('text'):
                        raise ValueError("Empty content in Gemini response")

                    return content[0]['text']

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        await asyncio.sleep(65)
                        retry_count += 1
                        continue
                    raise Exception(f"Gemini HTTP {e.response.status_code}: {e.response.text}")

            raise Exception("Max retries exceeded for Gemini API")
