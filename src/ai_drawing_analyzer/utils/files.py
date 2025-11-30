import os
import httpx
from urllib.parse import urlparse
from .logging import logger

def is_url(path: str) -> bool:
    return path.startswith('http://') or path.startswith('https://')

def detect_and_download(url_or_path: str, output_dir: str = "tmp_downloads") -> str:
    """Detect if input is a URL and download it."""
    if os.path.exists(url_or_path):
        return url_or_path
    
    if not is_url(url_or_path):
        return url_or_path
    
    os.makedirs(output_dir, exist_ok=True)
    return download_from_url(url_or_path, output_dir)

def download_from_url(url: str, output_dir: str) -> str:
    """Download file from direct URL"""
    try:
        filename = os.path.basename(urlparse(url).path) or 'downloaded_file.pdf'
        output_path = os.path.join(output_dir, filename)

        # Check if already downloaded
        if os.path.exists(output_path):
            logger.info(f"Using cached file: {filename}")
            return output_path

        logger.info(f"📥 Downloading: {url}")
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            response = client.get(url)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

        logger.info(f"✅ Downloaded: {filename}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise
