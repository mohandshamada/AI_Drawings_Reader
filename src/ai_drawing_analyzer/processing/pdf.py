import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from typing import Optional

class PDFProcessor:
    """Process PDF files and convert pages to high-resolution images"""

    def __init__(self, pdf_path: str, zoom: int = 2, quality: int = 90):
        """
        Initialize PDF processor.

        Args:
            pdf_path: Path to PDF file
            zoom: Zoom factor for page rendering (default 2x)
            quality: JPEG quality 1-100 (default 90)
        """
        if zoom < 1 or zoom > 4:
            raise ValueError("zoom must be between 1 and 4")
        if quality < 1 or quality > 100:
            raise ValueError("quality must be between 1 and 100")

        self.pdf_path = pdf_path
        self.zoom = zoom
        self.quality = quality
        self.doc = fitz.open(pdf_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.doc.close()

    def get_page_count(self) -> int:
        """Get total number of pages in PDF"""
        return len(self.doc)

    def get_page_as_image_base64(self, page_num: int) -> str:
        """
        Get page as base64 encoded JPEG image.

        Args:
            page_num: Zero-indexed page number

        Returns:
            Base64 encoded JPEG image string
        """
        if page_num < 0 or page_num >= len(self.doc):
            raise ValueError(f"Invalid page number: {page_num}")

        page = self.doc[page_num]

        # Matrix=zoom means zoom factor (higher resolution)
        pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=self.quality)
        img_data = img_byte_arr.getvalue()
        return base64.b64encode(img_data).decode('utf-8')
