"""Pytest configuration and fixtures"""
import pytest
import base64
from PIL import Image
import io


@pytest.fixture
def sample_image_base64():
    """Generate a simple test image in base64 format"""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_data = img_byte_arr.getvalue()
    return base64.b64encode(img_data).decode('utf-8')


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing"""
    return "test_api_key_12345"
