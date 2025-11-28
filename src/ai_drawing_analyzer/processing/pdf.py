import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from typing import Optional

class PDFProcessor:
    """Process PDF files and detect text vs image pages"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.doc.close()
    
    def get_page_count(self) -> int:
        return len(self.doc)
    
    def get_page_as_image_base64(self, page_num: int, zoom: int = 2) -> str:
        """Get page as base64 encoded image (JPEG for better compression)"""
        page = self.doc[page_num]
        
        # Matrix=zoom means zoom factor (higher resolution)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=90)
        img_data = img_byte_arr.getvalue()
        return base64.b64encode(img_data).decode('utf-8')
