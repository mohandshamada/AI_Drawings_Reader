"""Tests for IFC file processing"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestIFCProcessor:
    """Test IFC processor functionality"""

    def test_import_without_ifcopenshell(self):
        """Test that IFC processor handles missing ifcopenshell gracefully"""
        with patch.dict("sys.modules", {"ifcopenshell": None}):
            # Force reimport
            import importlib
            from ai_drawing_analyzer.processing import ifc

            importlib.reload(ifc)
            assert ifc.HAS_IFC is False

    def test_bounding_box_intersection(self):
        """Test bounding box intersection detection"""
        from ai_drawing_analyzer.processing.ifc import BoundingBox

        box1 = BoundingBox(min_x=0, min_y=0, min_z=0, max_x=2, max_y=2, max_z=2)
        box2 = BoundingBox(min_x=1, min_y=1, min_z=1, max_x=3, max_y=3, max_z=3)
        box3 = BoundingBox(min_x=5, min_y=5, min_z=5, max_x=6, max_y=6, max_z=6)

        # box1 and box2 should intersect
        assert box1.intersects(box2) is True
        # box1 and box3 should not intersect
        assert box1.intersects(box3) is False
        # box2 and box3 should not intersect
        assert box2.intersects(box3) is False

    def test_bounding_box_volume(self):
        """Test bounding box volume calculation"""
        from ai_drawing_analyzer.processing.ifc import BoundingBox

        box = BoundingBox(min_x=0, min_y=0, min_z=0, max_x=2, max_y=3, max_z=4)
        assert box.volume() == 24.0  # 2 * 3 * 4

    def test_bounding_box_intersection_volume(self):
        """Test intersection volume calculation"""
        from ai_drawing_analyzer.processing.ifc import BoundingBox

        box1 = BoundingBox(min_x=0, min_y=0, min_z=0, max_x=2, max_y=2, max_z=2)
        box2 = BoundingBox(min_x=1, min_y=1, min_z=1, max_x=3, max_y=3, max_z=3)

        # Intersection is a 1x1x1 cube
        assert box1.intersection_volume(box2) == 1.0

    def test_ifc_element_to_dict(self):
        """Test IFC element serialization"""
        from ai_drawing_analyzer.processing.ifc import IFCElement, BoundingBox

        elem = IFCElement(
            global_id="test123",
            ifc_type="IfcWall",
            name="Test Wall",
            description="A test wall",
            bounding_box=BoundingBox(0, 0, 0, 1, 1, 1),
        )

        result = elem.to_dict()
        assert result["global_id"] == "test123"
        assert result["ifc_type"] == "IfcWall"
        assert result["name"] == "Test Wall"
        assert "bounding_box" in result

    def test_clash_info_to_dict(self):
        """Test clash info serialization"""
        from ai_drawing_analyzer.processing.ifc import ClashInfo

        clash = ClashInfo(
            element1_id="elem1",
            element1_type="IfcPipe",
            element1_name="Pipe 1",
            element2_id="elem2",
            element2_type="IfcBeam",
            element2_name="Beam 1",
            clash_type="hard",
            severity="critical",
            intersection_volume=0.5,
            location=(1.0, 2.0, 3.0),
            description="Pipe penetrates beam",
        )

        result = clash.to_dict()
        assert result["clash_type"] == "hard"
        assert result["severity"] == "critical"
        assert result["intersection_volume"] == 0.5

    def test_issue_info_to_dict(self):
        """Test issue info serialization"""
        from ai_drawing_analyzer.processing.ifc import IssueInfo

        issue = IssueInfo(
            issue_type="missing_material",
            severity="minor",
            element_id="elem1",
            element_type="IfcWall",
            element_name="Wall 1",
            description="Wall has no material",
            suggestion="Assign material",
        )

        result = issue.to_dict()
        assert result["issue_type"] == "missing_material"
        assert result["severity"] == "minor"


class TestIFCConverter:
    """Test IFC converter functionality"""

    def test_converter_availability_check(self):
        """Test converter availability check"""
        from ai_drawing_analyzer.converters.ifc_converter import IFCConverter

        converter = IFCConverter()
        # Should return True or False based on ifcopenshell availability
        result = converter.is_available()
        assert isinstance(result, bool)

    def test_generate_recommendations_with_clashes(self):
        """Test recommendation generation for clashes"""
        from ai_drawing_analyzer.converters.ifc_converter import IFCConverter

        converter = IFCConverter()

        # Mock analysis with critical clashes
        analysis = {
            "statistics": {
                "clashes_by_severity": {"critical": 5, "major": 10},
                "issues_by_severity": {},
            },
            "clashes": [],
            "issues": [],
        }

        recommendations = converter._generate_recommendations(analysis)
        assert len(recommendations) >= 1
        assert any("URGENT" in r or "critical" in r.lower() for r in recommendations)

    def test_generate_recommendations_no_issues(self):
        """Test recommendation generation when no issues"""
        from ai_drawing_analyzer.converters.ifc_converter import IFCConverter

        converter = IFCConverter()

        analysis = {
            "statistics": {"clashes_by_severity": {}, "issues_by_severity": {}},
            "clashes": [],
            "issues": [],
        }

        recommendations = converter._generate_recommendations(analysis)
        assert len(recommendations) == 1
        assert "well-coordinated" in recommendations[0].lower()
