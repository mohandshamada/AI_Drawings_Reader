"""Tests for DWG/DXF file processing"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestDWGProcessor:
    """Test DWG processor functionality"""

    def test_import_without_ezdxf(self):
        """Test that DWG processor handles missing ezdxf gracefully"""
        with patch.dict("sys.modules", {"ezdxf": None}):
            import importlib
            from ai_drawing_analyzer.processing import dwg

            importlib.reload(dwg)
            assert dwg.HAS_EZDXF is False

    def test_cad_entity_to_dict(self):
        """Test CAD entity serialization"""
        from ai_drawing_analyzer.processing.dwg import CADEntity

        entity = CADEntity(
            handle="ABC123",
            entity_type="LINE",
            layer="A-WALL",
            color=7,
            geometry={"type": "line", "start": {"x": 0, "y": 0}, "end": {"x": 10, "y": 10}},
        )

        result = entity.to_dict()
        assert result["handle"] == "ABC123"
        assert result["entity_type"] == "LINE"
        assert result["layer"] == "A-WALL"
        assert result["geometry"]["type"] == "line"

    def test_cad_layer_to_dict(self):
        """Test CAD layer serialization"""
        from ai_drawing_analyzer.processing.dwg import CADLayer

        layer = CADLayer(
            name="A-WALL",
            color=7,
            linetype="Continuous",
            is_on=True,
            is_frozen=False,
            is_locked=False,
            entity_count=100,
        )

        result = layer.to_dict()
        assert result["name"] == "A-WALL"
        assert result["entity_count"] == 100
        assert result["is_on"] is True

    def test_cad_block_to_dict(self):
        """Test CAD block serialization"""
        from ai_drawing_analyzer.processing.dwg import CADBlock

        block = CADBlock(
            name="TITLE_BLOCK",
            base_point=(0, 0, 0),
            entity_count=50,
            insert_count=5,
            attributes=[{"tag": "PROJECT", "prompt": "Project Name"}],
        )

        result = block.to_dict()
        assert result["name"] == "TITLE_BLOCK"
        assert result["insert_count"] == 5
        assert len(result["attributes"]) == 1

    def test_cad_issue_to_dict(self):
        """Test CAD issue serialization"""
        from ai_drawing_analyzer.processing.dwg import CADIssue

        issue = CADIssue(
            issue_type="duplicate_entity",
            severity="major",
            entity_handle="ABC123",
            entity_type="LINE",
            layer="0",
            description="Duplicate line detected",
            suggestion="Delete duplicate",
            location=(10, 20, 0),
        )

        result = issue.to_dict()
        assert result["issue_type"] == "duplicate_entity"
        assert result["severity"] == "major"
        assert result["location"] == (10, 20, 0)


class TestDWGProcessorDisciplineDetection:
    """Test layer discipline detection"""

    def test_architecture_layer_detection(self):
        """Test detection of architectural layers"""
        from ai_drawing_analyzer.processing.dwg import DWGProcessor

        # Create a processor mock to test the method
        processor = DWGProcessor.__new__(DWGProcessor)

        assert processor._get_discipline("A-WALL") == "Architecture"
        assert processor._get_discipline("A-DOOR") == "Architecture"
        assert processor._get_discipline("ARCH-DETAIL") == "Architecture"

    def test_structure_layer_detection(self):
        """Test detection of structural layers"""
        from ai_drawing_analyzer.processing.dwg import DWGProcessor

        processor = DWGProcessor.__new__(DWGProcessor)

        assert processor._get_discipline("S-BEAM") == "Structure"
        assert processor._get_discipline("STRU-COLUMN") == "Structure"

    def test_mep_layer_detection(self):
        """Test detection of MEP layers"""
        from ai_drawing_analyzer.processing.dwg import DWGProcessor

        processor = DWGProcessor.__new__(DWGProcessor)

        assert processor._get_discipline("M-DUCT") == "Mechanical"
        assert processor._get_discipline("E-LITE") == "Electrical"
        assert processor._get_discipline("P-PIPE") == "Plumbing"

    def test_unknown_layer_discipline(self):
        """Test unknown layer discipline"""
        from ai_drawing_analyzer.processing.dwg import DWGProcessor

        processor = DWGProcessor.__new__(DWGProcessor)

        assert processor._get_discipline("RANDOM_LAYER") == "Other"


class TestDWGConverter:
    """Test DWG converter functionality"""

    def test_converter_availability_check(self):
        """Test converter availability check"""
        from ai_drawing_analyzer.converters.dwg_converter import DWGConverter

        converter = DWGConverter()
        result = converter.is_available()
        assert isinstance(result, bool)

    def test_generate_recommendations_with_issues(self):
        """Test recommendation generation for issues"""
        from ai_drawing_analyzer.converters.dwg_converter import DWGConverter

        converter = DWGConverter()

        # Mock analysis with issues
        analysis = {
            "statistics": {"issues_by_severity": {"major": 5, "warning": 10}},
            "issues": [
                {"issue_type": "duplicate_entity"} for _ in range(15)
            ] + [
                {"issue_type": "entity_on_layer_0"} for _ in range(25)
            ],
        }

        recommendations = converter._generate_recommendations(analysis)
        assert len(recommendations) >= 1
        assert any("duplicate" in r.lower() or "cleanup" in r.lower() for r in recommendations)

    def test_generate_recommendations_clean_drawing(self):
        """Test recommendation generation for clean drawing"""
        from ai_drawing_analyzer.converters.dwg_converter import DWGConverter

        converter = DWGConverter()

        analysis = {"statistics": {"issues_by_severity": {}}, "issues": []}

        recommendations = converter._generate_recommendations(analysis)
        assert len(recommendations) == 1
        assert "well-organized" in recommendations[0].lower()


class TestFileTypeDetection:
    """Test file type detection"""

    def test_pdf_detection(self):
        """Test PDF file detection"""
        from ai_drawing_analyzer.cli import get_file_type

        assert get_file_type("document.pdf") == "pdf"
        assert get_file_type("DOCUMENT.PDF") == "pdf"

    def test_ifc_detection(self):
        """Test IFC file detection"""
        from ai_drawing_analyzer.cli import get_file_type

        assert get_file_type("model.ifc") == "ifc"
        assert get_file_type("MODEL.IFC") == "ifc"

    def test_cad_detection(self):
        """Test CAD file detection"""
        from ai_drawing_analyzer.cli import get_file_type

        assert get_file_type("drawing.dwg") == "cad"
        assert get_file_type("drawing.dxf") == "cad"
        assert get_file_type("DRAWING.DWG") == "cad"

    def test_jsonl_detection(self):
        """Test JSONL file detection"""
        from ai_drawing_analyzer.cli import get_file_type

        assert get_file_type("output.jsonl") == "jsonl"
        assert get_file_type("output.json") == "jsonl"

    def test_unknown_file_type(self):
        """Test unknown file type"""
        from ai_drawing_analyzer.cli import get_file_type

        assert get_file_type("unknown.xyz") == "unknown"
