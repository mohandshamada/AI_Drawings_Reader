import os
from typing import Dict, Any

class AppConfig:
    """Application configuration with defaults"""
    # PDF Processing
    PDF_ZOOM_LEVEL = 2  # 2x zoom for high-resolution conversion
    JPEG_QUALITY = 90  # JPEG quality (1-100)

    # API & Network
    API_TIMEOUT = 120.0  # seconds
    DOWNLOAD_TIMEOUT = 60  # seconds
    MAX_RETRIES = 3
    RETRY_BACKOFF_MULTIPLIER = 1  # exponential backoff: base * multiplier^attempt

    # Processing
    MAX_TOKENS = 2048  # Max tokens in LLM response
    BATCH_SIZE = 1  # Pages processed at once (1 = sequential)

    # Output
    ENABLE_COMPRESSION = False  # Compress output JSON
    CACHE_DOWNLOADS = True  # Cache downloaded PDFs

    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """Load config from environment variables"""
        return {
            'pdf_zoom_level': int(os.getenv('PDF_ZOOM_LEVEL', cls.PDF_ZOOM_LEVEL)),
            'jpeg_quality': int(os.getenv('JPEG_QUALITY', cls.JPEG_QUALITY)),
            'api_timeout': float(os.getenv('API_TIMEOUT', cls.API_TIMEOUT)),
            'download_timeout': float(os.getenv('DOWNLOAD_TIMEOUT', cls.DOWNLOAD_TIMEOUT)),
            'max_retries': int(os.getenv('MAX_RETRIES', cls.MAX_RETRIES)),
            'max_tokens': int(os.getenv('MAX_TOKENS', cls.MAX_TOKENS)),
            'batch_size': int(os.getenv('BATCH_SIZE', cls.BATCH_SIZE)),
        }

def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file"""
    env_vars = {}
    if not os.path.exists(env_path):
        return env_vars

    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        env_vars[key] = value
                        # Also set in os.environ if not present
                        if key not in os.environ:
                            os.environ[key] = value
    except Exception as e:
        print(f"⚠️ Warning: Could not read {env_path}: {e}")

    return env_vars
