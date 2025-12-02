"""Process IFC (Industry Foundation Classes) files and extract BIM data for LLM analysis"""

import json
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as element_util
    import ifcopenshell.util.placement as placement_util

    HAS_IFC = True
except ImportError:
    HAS_IFC = False


@dataclass
class BoundingBox:
    """3D bounding box for spatial analysis"""

    min_x: float = 0.0
    min_y: float = 0.0
    min_z: float = 0.0
    max_x: float = 0.0
    max_y: float = 0.0
    max_z: float = 0.0

    def intersects(self, other: "BoundingBox", tolerance: float = 0.001) -> bool:
        """Check if two bounding boxes intersect (potential clash)"""
        return not (
            self.max_x < other.min_x - tolerance
            or self.min_x > other.max_x + tolerance
            or self.max_y < other.min_y - tolerance
            or self.min_y > other.max_y + tolerance
            or self.max_z < other.min_z - tolerance
            or self.min_z > other.max_z + tolerance
        )

    def volume(self) -> float:
        """Calculate bounding box volume"""
        return (
            (self.max_x - self.min_x)
            * (self.max_y - self.min_y)
            * (self.max_z - self.min_z)
        )

    def intersection_volume(self, other: "BoundingBox") -> float:
        """Calculate intersection volume between two boxes"""
        if not self.intersects(other):
            return 0.0

        ix_min = max(self.min_x, other.min_x)
        iy_min = max(self.min_y, other.min_y)
        iz_min = max(self.min_z, other.min_z)
        ix_max = min(self.max_x, other.max_x)
        iy_max = min(self.max_y, other.max_y)
        iz_max = min(self.max_z, other.max_z)

        return max(0, ix_max - ix_min) * max(0, iy_max - iy_min) * max(0, iz_max - iz_min)


@dataclass
class IFCElement:
    """Represents a single IFC element with all relevant properties"""

    global_id: str
    ifc_type: str
    name: str
    description: str = ""
    object_type: str = ""
    material: str = ""
    layer: str = ""
    level: str = ""
    system: str = ""
    classification: str = ""
    properties: dict = field(default_factory=dict)
    quantities: dict = field(default_factory=dict)
    bounding_box: Optional[BoundingBox] = None
    location: tuple = (0.0, 0.0, 0.0)
    relationships: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        if self.bounding_box:
            result["bounding_box"] = asdict(self.bounding_box)
        return result


@dataclass
class ClashInfo:
    """Information about a detected clash between elements"""

    element1_id: str
    element1_type: str
    element1_name: str
    element2_id: str
    element2_type: str
    element2_name: str
    clash_type: str  # "hard", "soft", "clearance"
    severity: str  # "critical", "major", "minor"
    intersection_volume: float
    location: tuple
    description: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class IssueInfo:
    """Information about a detected issue in the model"""

    issue_type: str
    severity: str  # "critical", "major", "minor", "warning"
    element_id: str
    element_type: str
    element_name: str
    description: str
    suggestion: str
    location: Optional[tuple] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class IFCProcessor:
    """Process IFC files and extract structured data for LLM analysis"""

    # Element types that are typically involved in clashes
    CLASH_RELEVANT_TYPES = {
        "IfcWall",
        "IfcWallStandardCase",
        "IfcColumn",
        "IfcBeam",
        "IfcSlab",
        "IfcRoof",
        "IfcStair",
        "IfcRamp",
        "IfcDoor",
        "IfcWindow",
        "IfcPipe",
        "IfcPipeSegment",
        "IfcPipeFitting",
        "IfcDuctSegment",
        "IfcDuctFitting",
        "IfcCableCarrierSegment",
        "IfcCableSegment",
        "IfcFlowTerminal",
        "IfcFlowSegment",
        "IfcFlowFitting",
        "IfcBuildingElementProxy",
        "IfcFurnishingElement",
        "IfcMember",
        "IfcPlate",
        "IfcFooting",
        "IfcPile",
        "IfcCovering",
        "IfcCurtainWall",
        "IfcRailing",
    }

    # Disciplines based on element types
    DISCIPLINE_MAPPING = {
        "Architecture": {
            "IfcWall",
            "IfcWallStandardCase",
            "IfcDoor",
            "IfcWindow",
            "IfcSlab",
            "IfcRoof",
            "IfcStair",
            "IfcRamp",
            "IfcCurtainWall",
            "IfcCovering",
            "IfcRailing",
        },
        "Structure": {
            "IfcColumn",
            "IfcBeam",
            "IfcFooting",
            "IfcPile",
            "IfcMember",
            "IfcPlate",
            "IfcReinforcingBar",
            "IfcReinforcingMesh",
            "IfcTendon",
        },
        "MEP_Mechanical": {
            "IfcDuctSegment",
            "IfcDuctFitting",
            "IfcAirTerminal",
            "IfcFan",
            "IfcAirTerminalBox",
            "IfcDamper",
            "IfcHeatExchanger",
            "IfcHumidifier",
            "IfcEvaporativeCooler",
            "IfcCoil",
        },
        "MEP_Plumbing": {
            "IfcPipe",
            "IfcPipeSegment",
            "IfcPipeFitting",
            "IfcValve",
            "IfcPump",
            "IfcTank",
            "IfcSanitaryTerminal",
            "IfcWasteTerminal",
            "IfcStackTerminal",
        },
        "MEP_Electrical": {
            "IfcCableCarrierSegment",
            "IfcCableSegment",
            "IfcCableCarrierFitting",
            "IfcCableFitting",
            "IfcElectricDistributionBoard",
            "IfcElectricMotor",
            "IfcLightFixture",
            "IfcOutlet",
            "IfcSwitchingDevice",
        },
        "MEP_FireProtection": {
            "IfcFireSuppressionTerminal",
            "IfcInterceptor",
        },
    }

    def __init__(self, ifc_path: str):
        """
        Initialize IFC processor.

        Args:
            ifc_path: Path to IFC file

        Raises:
            ImportError: If ifcopenshell is not installed
            FileNotFoundError: If IFC file doesn't exist
        """
        if not HAS_IFC:
            raise ImportError(
                "ifcopenshell is not installed. Install with: pip install ifcopenshell"
            )

        self.ifc_path = Path(ifc_path)
        if not self.ifc_path.exists():
            raise FileNotFoundError(f"IFC file not found: {ifc_path}")

        self.model = ifcopenshell.open(str(self.ifc_path))
        self.elements: list[IFCElement] = []
        self.clashes: list[ClashInfo] = []
        self.issues: list[IssueInfo] = []
        self._geometry_settings = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ifcopenshell doesn't require explicit cleanup but we keep the pattern
        self.model = None

    def get_project_info(self) -> dict:
        """Extract project-level information"""
        project = self.model.by_type("IfcProject")
        if not project:
            return {}

        project = project[0]
        info = {
            "name": project.Name or "Unnamed Project",
            "description": project.Description or "",
            "phase": project.Phase or "",
            "units": self._get_units(),
            "schema": self.model.schema,
            "file_name": self.ifc_path.name,
        }

        # Get site and building info
        sites = self.model.by_type("IfcSite")
        if sites:
            site = sites[0]
            info["site"] = {
                "name": site.Name or "",
                "description": site.Description or "",
            }

        buildings = self.model.by_type("IfcBuilding")
        if buildings:
            building = buildings[0]
            info["building"] = {
                "name": building.Name or "",
                "description": building.Description or "",
            }

        # Count elements by type
        info["element_summary"] = self._get_element_summary()

        return info

    def _get_units(self) -> dict:
        """Extract unit information from the model"""
        units = {}
        try:
            unit_assignments = self.model.by_type("IfcUnitAssignment")
            if unit_assignments:
                for unit in unit_assignments[0].Units:
                    if hasattr(unit, "UnitType"):
                        unit_name = unit.UnitType
                        if hasattr(unit, "Name"):
                            units[unit_name] = unit.Name
        except Exception:
            pass
        return units

    def _get_element_summary(self) -> dict:
        """Get count of elements by type"""
        summary = defaultdict(int)
        for element in self.model.by_type("IfcElement"):
            summary[element.is_a()] += 1
        return dict(summary)

    def _get_discipline(self, ifc_type: str) -> str:
        """Determine discipline based on IFC type"""
        for discipline, types in self.DISCIPLINE_MAPPING.items():
            if ifc_type in types:
                return discipline
        return "Other"

    def _get_element_properties(self, element) -> dict:
        """Extract all properties from an element"""
        properties = {}
        try:
            psets = element_util.get_psets(element)
            for pset_name, pset_props in psets.items():
                for prop_name, prop_value in pset_props.items():
                    if prop_value is not None:
                        key = f"{pset_name}.{prop_name}"
                        properties[key] = str(prop_value)
        except Exception:
            pass
        return properties

    def _get_element_quantities(self, element) -> dict:
        """Extract quantities from an element"""
        quantities = {}
        try:
            qtos = element_util.get_psets(element, qtos_only=True)
            for qto_name, qto_props in qtos.items():
                for prop_name, prop_value in qto_props.items():
                    if prop_value is not None:
                        key = f"{qto_name}.{prop_name}"
                        quantities[key] = prop_value
        except Exception:
            pass
        return quantities

    def _get_element_material(self, element) -> str:
        """Get material name for an element"""
        try:
            material = element_util.get_material(element)
            if material:
                if hasattr(material, "Name"):
                    return material.Name or ""
                elif hasattr(material, "ForLayerSet"):
                    # LayerSetUsage
                    layer_set = material.ForLayerSet
                    if layer_set and hasattr(layer_set, "LayerSetName"):
                        return layer_set.LayerSetName or ""
        except Exception:
            pass
        return ""

    def _get_element_level(self, element) -> str:
        """Get the building storey/level for an element"""
        try:
            container = element_util.get_container(element)
            if container and container.is_a("IfcBuildingStorey"):
                return container.Name or ""
        except Exception:
            pass
        return ""

    def _get_bounding_box(self, element) -> Optional[BoundingBox]:
        """Calculate bounding box for an element"""
        try:
            if self._geometry_settings is None:
                self._geometry_settings = ifcopenshell.geom.settings()
                self._geometry_settings.set(
                    self._geometry_settings.USE_WORLD_COORDS, True
                )

            shape = ifcopenshell.geom.create_shape(self._geometry_settings, element)
            if shape:
                verts = shape.geometry.verts
                if verts:
                    xs = verts[0::3]
                    ys = verts[1::3]
                    zs = verts[2::3]
                    return BoundingBox(
                        min_x=min(xs),
                        min_y=min(ys),
                        min_z=min(zs),
                        max_x=max(xs),
                        max_y=max(ys),
                        max_z=max(zs),
                    )
        except Exception:
            # Geometry processing may fail for some elements
            pass
        return None

    def _get_element_location(self, element) -> tuple:
        """Get the placement location of an element"""
        try:
            placement = element.ObjectPlacement
            if placement:
                matrix = placement_util.get_local_placement(placement)
                if matrix is not None:
                    return (float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]))
        except Exception:
            pass
        return (0.0, 0.0, 0.0)

    def extract_elements(self, include_geometry: bool = True) -> list[IFCElement]:
        """
        Extract all relevant elements from the IFC file.

        Args:
            include_geometry: Whether to compute bounding boxes (slower but needed for clash detection)

        Returns:
            List of IFCElement objects
        """
        self.elements = []

        for element in self.model.by_type("IfcElement"):
            ifc_type = element.is_a()

            ifc_elem = IFCElement(
                global_id=element.GlobalId,
                ifc_type=ifc_type,
                name=element.Name or "",
                description=element.Description or "",
                object_type=element.ObjectType or "",
                material=self._get_element_material(element),
                level=self._get_element_level(element),
                properties=self._get_element_properties(element),
                quantities=self._get_element_quantities(element),
                location=self._get_element_location(element),
            )

            # Get system/classification info
            try:
                if hasattr(element, "HasAssociations"):
                    for assoc in element.HasAssociations:
                        if assoc.is_a("IfcRelAssociatesClassification"):
                            ref = assoc.RelatingClassification
                            if hasattr(ref, "ItemReference"):
                                ifc_elem.classification = ref.ItemReference or ""
            except Exception:
                pass

            if include_geometry and ifc_type in self.CLASH_RELEVANT_TYPES:
                ifc_elem.bounding_box = self._get_bounding_box(element)

            self.elements.append(ifc_elem)

        return self.elements

    def detect_clashes(self, tolerance: float = 0.01) -> list[ClashInfo]:
        """
        Detect geometric clashes between elements.

        Args:
            tolerance: Distance tolerance for clash detection (in model units)

        Returns:
            List of ClashInfo objects describing detected clashes
        """
        if not self.elements:
            self.extract_elements(include_geometry=True)

        self.clashes = []

        # Filter elements that have bounding boxes and are clash-relevant
        clash_elements = [
            e
            for e in self.elements
            if e.bounding_box is not None and e.ifc_type in self.CLASH_RELEVANT_TYPES
        ]

        # Simple O(n^2) clash detection using bounding boxes
        # For production, use spatial indexing (R-tree)
        for i, elem1 in enumerate(clash_elements):
            for elem2 in clash_elements[i + 1 :]:
                if elem1.bounding_box.intersects(elem2.bounding_box, tolerance):
                    # Skip same-level architectural elements that commonly touch
                    if elem1.level == elem2.level:
                        same_discipline = self._get_discipline(
                            elem1.ifc_type
                        ) == self._get_discipline(elem2.ifc_type)
                        if same_discipline and self._get_discipline(elem1.ifc_type) in [
                            "Architecture",
                            "Structure",
                        ]:
                            # Check for significant overlap, not just touching
                            intersection = elem1.bounding_box.intersection_volume(
                                elem2.bounding_box
                            )
                            min_volume = min(
                                elem1.bounding_box.volume(), elem2.bounding_box.volume()
                            )
                            if min_volume > 0 and intersection / min_volume < 0.1:
                                continue

                    intersection_volume = elem1.bounding_box.intersection_volume(
                        elem2.bounding_box
                    )

                    # Determine clash type and severity
                    clash_type = self._determine_clash_type(elem1, elem2)
                    severity = self._determine_clash_severity(
                        elem1, elem2, intersection_volume
                    )

                    # Calculate approximate clash location
                    location = (
                        (elem1.location[0] + elem2.location[0]) / 2,
                        (elem1.location[1] + elem2.location[1]) / 2,
                        (elem1.location[2] + elem2.location[2]) / 2,
                    )

                    clash = ClashInfo(
                        element1_id=elem1.global_id,
                        element1_type=elem1.ifc_type,
                        element1_name=elem1.name,
                        element2_id=elem2.global_id,
                        element2_type=elem2.ifc_type,
                        element2_name=elem2.name,
                        clash_type=clash_type,
                        severity=severity,
                        intersection_volume=intersection_volume,
                        location=location,
                        description=self._generate_clash_description(elem1, elem2, clash_type),
                    )
                    self.clashes.append(clash)

        return self.clashes

    def _determine_clash_type(self, elem1: IFCElement, elem2: IFCElement) -> str:
        """Determine the type of clash between two elements"""
        disc1 = self._get_discipline(elem1.ifc_type)
        disc2 = self._get_discipline(elem2.ifc_type)

        # MEP vs Structure/Architecture is usually hard clash
        if ("MEP" in disc1 or "MEP" in disc2) and (
            disc1 in ["Structure", "Architecture"] or disc2 in ["Structure", "Architecture"]
        ):
            return "hard"

        # MEP vs MEP can be hard clash
        if "MEP" in disc1 and "MEP" in disc2:
            return "hard"

        # Structure vs Architecture
        if {disc1, disc2} == {"Structure", "Architecture"}:
            return "hard"

        return "soft"

    def _determine_clash_severity(
        self, elem1: IFCElement, elem2: IFCElement, intersection_volume: float
    ) -> str:
        """Determine severity of a clash"""
        disc1 = self._get_discipline(elem1.ifc_type)
        disc2 = self._get_discipline(elem2.ifc_type)

        # Structural clashes are critical
        if "Structure" in [disc1, disc2]:
            return "critical"

        # Large intersection volumes are major
        if intersection_volume > 0.1:  # 0.1 cubic meters
            return "major"

        # MEP clashes are usually major
        if "MEP" in disc1 or "MEP" in disc2:
            return "major"

        return "minor"

    def _generate_clash_description(
        self, elem1: IFCElement, elem2: IFCElement, clash_type: str
    ) -> str:
        """Generate a human-readable clash description"""
        disc1 = self._get_discipline(elem1.ifc_type)
        disc2 = self._get_discipline(elem2.ifc_type)

        name1 = elem1.name or elem1.ifc_type
        name2 = elem2.name or elem2.ifc_type
        level = elem1.level or elem2.level or "Unknown level"

        return f"{clash_type.capitalize()} clash between {disc1} element '{name1}' and {disc2} element '{name2}' at {level}"

    def detect_issues(self) -> list[IssueInfo]:
        """
        Detect common modeling issues in the IFC file.

        Returns:
            List of IssueInfo objects describing detected issues
        """
        if not self.elements:
            self.extract_elements(include_geometry=False)

        self.issues = []

        for elem in self.elements:
            # Check for missing names
            if not elem.name:
                self.issues.append(
                    IssueInfo(
                        issue_type="missing_name",
                        severity="warning",
                        element_id=elem.global_id,
                        element_type=elem.ifc_type,
                        element_name=f"<unnamed {elem.ifc_type}>",
                        description=f"Element of type {elem.ifc_type} has no name assigned",
                        suggestion="Add a descriptive name to improve model clarity and searchability",
                        location=elem.location,
                    )
                )

            # Check for missing material
            if not elem.material and elem.ifc_type in self.CLASH_RELEVANT_TYPES:
                self.issues.append(
                    IssueInfo(
                        issue_type="missing_material",
                        severity="minor",
                        element_id=elem.global_id,
                        element_type=elem.ifc_type,
                        element_name=elem.name or f"<unnamed {elem.ifc_type}>",
                        description=f"Element '{elem.name or elem.ifc_type}' has no material assigned",
                        suggestion="Assign appropriate material for accurate quantity takeoffs and visualization",
                        location=elem.location,
                    )
                )

            # Check for missing level assignment
            if not elem.level and elem.ifc_type in self.CLASH_RELEVANT_TYPES:
                self.issues.append(
                    IssueInfo(
                        issue_type="missing_level",
                        severity="major",
                        element_id=elem.global_id,
                        element_type=elem.ifc_type,
                        element_name=elem.name or f"<unnamed {elem.ifc_type}>",
                        description=f"Element '{elem.name or elem.ifc_type}' is not assigned to any building storey",
                        suggestion="Assign element to appropriate building storey for proper organization",
                        location=elem.location,
                    )
                )

            # Check for potential duplicate elements (same type, same location)
            if elem.bounding_box:
                for other_elem in self.elements:
                    if (
                        other_elem.global_id != elem.global_id
                        and other_elem.ifc_type == elem.ifc_type
                        and other_elem.bounding_box
                    ):
                        # Check if locations are very close (potential duplicates)
                        dist = (
                            (elem.location[0] - other_elem.location[0]) ** 2
                            + (elem.location[1] - other_elem.location[1]) ** 2
                            + (elem.location[2] - other_elem.location[2]) ** 2
                        ) ** 0.5

                        if dist < 0.001:  # Less than 1mm apart
                            # Avoid duplicate issues
                            if elem.global_id < other_elem.global_id:
                                self.issues.append(
                                    IssueInfo(
                                        issue_type="potential_duplicate",
                                        severity="major",
                                        element_id=elem.global_id,
                                        element_type=elem.ifc_type,
                                        element_name=elem.name
                                        or f"<unnamed {elem.ifc_type}>",
                                        description=f"Element '{elem.name or elem.ifc_type}' may be duplicated (another {elem.ifc_type} at same location)",
                                        suggestion="Review and remove duplicate element if unintended",
                                        location=elem.location,
                                    )
                                )

        return self.issues

    def to_json(
        self,
        output_path: Optional[str] = None,
        include_geometry: bool = True,
        detect_clashes_flag: bool = True,
        detect_issues_flag: bool = True,
    ) -> dict:
        """
        Convert IFC model to LLM-friendly JSON format.

        Args:
            output_path: Optional path to write JSON file
            include_geometry: Include bounding box information
            detect_clashes_flag: Run clash detection
            detect_issues_flag: Run issue detection

        Returns:
            Dictionary containing the full model analysis
        """
        # Extract elements if not already done
        if not self.elements:
            self.extract_elements(include_geometry=include_geometry)

        # Detect clashes and issues if requested
        if detect_clashes_flag:
            self.detect_clashes()
        if detect_issues_flag:
            self.detect_issues()

        # Build the JSON structure
        result = {
            "format_version": "1.0",
            "format_type": "ifc_analysis",
            "description": "IFC model analysis for LLM processing - includes elements, clashes, and issues",
            "project": self.get_project_info(),
            "statistics": {
                "total_elements": len(self.elements),
                "elements_by_type": self._get_element_summary(),
                "elements_by_discipline": self._count_by_discipline(),
                "total_clashes": len(self.clashes),
                "clashes_by_severity": self._count_clashes_by_severity(),
                "total_issues": len(self.issues),
                "issues_by_severity": self._count_issues_by_severity(),
            },
            "clashes": [c.to_dict() for c in self.clashes],
            "issues": [i.to_dict() for i in self.issues],
            "elements": [e.to_dict() for e in self.elements],
            "llm_analysis_hints": {
                "clash_priorities": [
                    "Focus on 'critical' and 'major' severity clashes first",
                    "Hard clashes between MEP and Structure require immediate attention",
                    "Check if clashing elements are on the same level/storey",
                    "Consider construction sequence when evaluating clashes",
                ],
                "issue_priorities": [
                    "Missing level assignments can cause scheduling problems",
                    "Duplicate elements affect quantity takeoffs",
                    "Missing materials affect cost estimation",
                ],
                "common_patterns": [
                    "Pipes penetrating beams often indicate coordination issues",
                    "Ducts through structural elements need sleeves or openings",
                    "Electrical conduits should maintain clearance from water pipes",
                ],
            },
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    def _count_by_discipline(self) -> dict:
        """Count elements by discipline"""
        counts = defaultdict(int)
        for elem in self.elements:
            discipline = self._get_discipline(elem.ifc_type)
            counts[discipline] += 1
        return dict(counts)

    def _count_clashes_by_severity(self) -> dict:
        """Count clashes by severity"""
        counts = defaultdict(int)
        for clash in self.clashes:
            counts[clash.severity] += 1
        return dict(counts)

    def _count_issues_by_severity(self) -> dict:
        """Count issues by severity"""
        counts = defaultdict(int)
        for issue in self.issues:
            counts[issue.severity] += 1
        return dict(counts)
