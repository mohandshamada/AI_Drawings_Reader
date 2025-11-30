"""Tests for PDF processing functionality"""
import pytest
from ai_drawing_analyzer.processing.pdf import PDFProcessor


def test_pdf_processor_zoom_validation():
    """Test that zoom parameter validation works correctly"""
    with pytest.raises(ValueError, match="zoom must be between 1 and 4"):
        PDFProcessor("test.pdf", zoom=5)

    with pytest.raises(ValueError, match="zoom must be between 1 and 4"):
        PDFProcessor("test.pdf", zoom=0)


def test_pdf_processor_quality_validation():
    """Test that quality parameter validation works correctly"""
    with pytest.raises(ValueError, match="quality must be between 1 and 100"):
        PDFProcessor("test.pdf", quality=101)

    with pytest.raises(ValueError, match="quality must be between 1 and 100"):
        PDFProcessor("test.pdf", quality=0)


def test_pdf_processor_valid_parameters():
    """Test that valid parameters don't raise errors on init"""
    # This will fail when trying to open the file, but validates params
    try:
        processor = PDFProcessor("test.pdf", zoom=2, quality=90)
    except Exception as e:
        # Should fail on file not found, not validation
        assert "zoom" not in str(e).lower()
        assert "quality" not in str(e).lower()
