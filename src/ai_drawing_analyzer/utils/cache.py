"""Response caching utilities for API calls"""
import hashlib
import json
from pathlib import Path
from typing import Optional
from .logging import logger


class ResponseCache:
    """Cache for AI model responses to avoid redundant API calls"""

    def __init__(self, cache_dir: Path = Path(".cache"), enabled: bool = True):
        """
        Initialize response cache.

        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled (default: True)
        """
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _get_cache_key(self, image_base64: str, prompt: str, model: str) -> str:
        """Generate cache key from image, prompt, and model"""
        # Use first 500 chars of base64 image to avoid huge keys
        content = f"{model}:{prompt}:{image_base64[:500]}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, image_base64: str, prompt: str, model: str) -> Optional[str]:
        """
        Get cached response if available.

        Args:
            image_base64: Base64 encoded image
            prompt: Text prompt
            model: Model identifier

        Returns:
            Cached response text if available, None otherwise
        """
        if not self.enabled:
            return None

        try:
            key = self._get_cache_key(image_base64, prompt, model)
            cache_file = self.cache_dir / f"{key}.json"

            if cache_file.exists():
                data = json.loads(cache_file.read_text())
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return data.get("response")
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def set(self, image_base64: str, prompt: str, model: str, response: str) -> None:
        """
        Cache a response.

        Args:
            image_base64: Base64 encoded image
            prompt: Text prompt
            model: Model identifier
            response: Response text to cache
        """
        if not self.enabled:
            return

        try:
            key = self._get_cache_key(image_base64, prompt, model)
            cache_file = self.cache_dir / f"{key}.json"

            cache_data = {
                "model": model,
                "prompt": prompt,
                "image_preview": image_base64[:100],
                "response": response
            }

            cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))
            logger.debug(f"Cached response for key: {key[:16]}...")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of cache files deleted
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cached responses")
        return count

    def size(self) -> int:
        """
        Get number of cached responses.

        Returns:
            Number of cache files
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        return len(list(self.cache_dir.glob("*.json")))
