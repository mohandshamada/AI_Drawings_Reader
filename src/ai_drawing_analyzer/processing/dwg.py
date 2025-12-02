"""Process DWG/DXF (CAD) files and extract structured data for LLM analysis"""

import json
import subprocess
import shutil
import tempfile
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import math

try:
    import ezdxf
    from ezdxf.addons import odafc
    from ezdxf.math import Vec3

    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False


@dataclass
class Point2D:
    """2D point representation"""

    x: float = 0.0
    y: float = 0.0


@dataclass
class Point3D:
    """3D point representation"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class CADEntity:
    """Represents a CAD entity with all relevant properties"""

    handle: str
    entity_type: str
    layer: str
    color: int = 256  # 256 = bylayer
    linetype: str = "BYLAYER"
    lineweight: int = -1  # -1 = bylayer
    geometry: dict = field(default_factory=dict)
    properties: dict = field(default_factory=dict)
    text_content: str = ""
    block_name: str = ""
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class CADLayer:
    """Represents a CAD layer"""

    name: str
    color: int = 7  # White/Black
    linetype: str = "Continuous"
    is_on: bool = True
    is_frozen: bool = False
    is_locked: bool = False
    entity_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class CADBlock:
    """Represents a CAD block definition"""

    name: str
    base_point: tuple = (0.0, 0.0, 0.0)
    entity_count: int = 0
    insert_count: int = 0  # How many times block is inserted
    attributes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class CADIssue:
    """Information about a detected issue in the CAD file"""

    issue_type: str
    severity: str  # "critical", "major", "minor", "warning"
    entity_handle: str
    entity_type: str
    layer: str
    description: str
    suggestion: str
    location: Optional[tuple] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class DWGProcessor:
    """Process DWG/DXF files and extract structured data for LLM analysis"""

    # Common layer name patterns and their likely disciplines
    LAYER_DISCIPLINE_PATTERNS = {
        "Architecture": ["A-", "ARCH", "WALL", "DOOR", "WIND", "FURN", "CEIL", "FLOOR", "ROOF"],
        "Structure": ["S-", "STRU", "BEAM", "COLM", "SLAB", "FOOT", "FNDN", "CONC", "STEEL"],
        "Mechanical": ["M-", "MECH", "HVAC", "DUCT", "DIFF", "VAV", "AHU"],
        "Electrical": ["E-", "ELEC", "LITE", "POWR", "FIRE", "COMM", "DATA"],
        "Plumbing": ["P-", "PLUM", "PIPE", "FIXT", "SANR", "DOMW"],
        "Civil": ["C-", "CIVL", "SITE", "TOPO", "ROAD", "PARK", "LAND"],
        "General": ["G-", "GENR", "ANNO", "TEXT", "DIM", "NOTE", "SYMB", "TITL", "BORD"],
    }

    def __init__(self, cad_path: str, oda_converter_path: Optional[str] = None):
        """
        Initialize CAD processor.

        Args:
            cad_path: Path to DWG or DXF file
            oda_converter_path: Path to ODA File Converter (optional, for DWG support)

        Raises:
            ImportError: If ezdxf is not installed
            FileNotFoundError: If CAD file doesn't exist
        """
        if not HAS_EZDXF:
            raise ImportError("ezdxf is not installed. Install with: pip install ezdxf")

        self.cad_path = Path(cad_path)
        if not self.cad_path.exists():
            raise FileNotFoundError(f"CAD file not found: {cad_path}")

        self.oda_converter_path = oda_converter_path
        self.doc = None
        self.entities: list[CADEntity] = []
        self.layers: list[CADLayer] = []
        self.blocks: list[CADBlock] = []
        self.issues: list[CADIssue] = []
        self._temp_dxf_path = None

        # Load the document
        self._load_document()

    def _load_document(self):
        """Load the CAD document, converting DWG to DXF if necessary"""
        file_ext = self.cad_path.suffix.lower()

        if file_ext == ".dxf":
            self.doc = ezdxf.readfile(str(self.cad_path))
        elif file_ext == ".dwg":
            # Try to convert DWG to DXF
            self._temp_dxf_path = self._convert_dwg_to_dxf()
            if self._temp_dxf_path:
                self.doc = ezdxf.readfile(self._temp_dxf_path)
            else:
                raise RuntimeError(
                    "Cannot read DWG file directly. Please install ODA File Converter "
                    "(https://www.opendesign.com/guestfiles/oda_file_converter) "
                    "or provide a DXF file instead."
                )
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def _convert_dwg_to_dxf(self) -> Optional[str]:
        """Convert DWG to DXF using ODA File Converter or ezdxf's odafc"""
        try:
            # Try using ezdxf's built-in ODA support
            if odafc.is_available():
                temp_dir = tempfile.mkdtemp()
                output_dxf = Path(temp_dir) / (self.cad_path.stem + ".dxf")
                odafc.convert(str(self.cad_path), str(output_dxf))
                return str(output_dxf)
        except Exception:
            pass

        # Try using ODA File Converter directly
        converter = self.oda_converter_path or self._find_oda_converter()
        if converter:
            try:
                temp_dir = tempfile.mkdtemp()
                subprocess.run(
                    [
                        converter,
                        str(self.cad_path.parent),  # Input folder
                        temp_dir,  # Output folder
                        "ACAD2018",  # Output version
                        "DXF",  # Output format
                        "0",  # Recurse subdirectories (0 = no)
                        "1",  # Audit each file (1 = yes)
                        self.cad_path.name,  # Filter (specific file)
                    ],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                output_dxf = Path(temp_dir) / (self.cad_path.stem + ".dxf")
                if output_dxf.exists():
                    return str(output_dxf)
            except Exception:
                pass

        return None

    def _find_oda_converter(self) -> Optional[str]:
        """Find ODA File Converter in common locations"""
        common_paths = [
            "/usr/bin/ODAFileConverter",
            "/usr/local/bin/ODAFileConverter",
            "/opt/ODAFileConverter/ODAFileConverter",
            "C:\\Program Files\\ODA\\ODAFileConverter\\ODAFileConverter.exe",
            "C:\\Program Files (x86)\\ODA\\ODAFileConverter\\ODAFileConverter.exe",
        ]

        for path in common_paths:
            if Path(path).exists():
                return path

        # Try to find in PATH
        converter = shutil.which("ODAFileConverter")
        return converter

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.doc = None
        # Cleanup temp files
        if self._temp_dxf_path:
            try:
                temp_dir = Path(self._temp_dxf_path).parent
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    def get_drawing_info(self) -> dict:
        """Extract drawing-level information"""
        info = {
            "file_name": self.cad_path.name,
            "file_format": self.cad_path.suffix.lower(),
            "dxf_version": self.doc.dxfversion if self.doc else "Unknown",
            "units": self._get_units(),
            "extents": self._get_extents(),
        }

        # Get header variables
        try:
            header = self.doc.header
            info["created_by"] = header.get("$LASTSAVEDBY", "Unknown")
            info["acadver"] = header.get("$ACADVER", "Unknown")
        except Exception:
            pass

        # Count entities
        info["entity_summary"] = self._get_entity_summary()

        return info

    def _get_units(self) -> dict:
        """Get drawing units"""
        units_map = {
            0: "Unitless",
            1: "Inches",
            2: "Feet",
            3: "Miles",
            4: "Millimeters",
            5: "Centimeters",
            6: "Meters",
            7: "Kilometers",
            8: "Microinches",
            9: "Mils",
            10: "Yards",
            11: "Angstroms",
            12: "Nanometers",
            13: "Microns",
            14: "Decimeters",
            15: "Decameters",
            16: "Hectometers",
            17: "Gigameters",
            18: "Astronomical units",
            19: "Light years",
            20: "Parsecs",
        }

        try:
            insunits = self.doc.header.get("$INSUNITS", 0)
            return {"code": insunits, "name": units_map.get(insunits, "Unknown")}
        except Exception:
            return {"code": 0, "name": "Unknown"}

    def _get_extents(self) -> dict:
        """Get drawing extents"""
        try:
            header = self.doc.header
            extmin = header.get("$EXTMIN", (0, 0, 0))
            extmax = header.get("$EXTMAX", (0, 0, 0))
            return {
                "min": {"x": extmin[0], "y": extmin[1], "z": extmin[2] if len(extmin) > 2 else 0},
                "max": {"x": extmax[0], "y": extmax[1], "z": extmax[2] if len(extmax) > 2 else 0},
            }
        except Exception:
            return {"min": {"x": 0, "y": 0, "z": 0}, "max": {"x": 0, "y": 0, "z": 0}}

    def _get_entity_summary(self) -> dict:
        """Get count of entities by type"""
        summary = defaultdict(int)
        try:
            msp = self.doc.modelspace()
            for entity in msp:
                summary[entity.dxftype()] += 1
        except Exception:
            pass
        return dict(summary)

    def _get_discipline(self, layer_name: str) -> str:
        """Determine discipline based on layer name"""
        layer_upper = layer_name.upper()
        for discipline, patterns in self.LAYER_DISCIPLINE_PATTERNS.items():
            for pattern in patterns:
                if pattern.upper() in layer_upper or layer_upper.startswith(pattern.upper()):
                    return discipline
        return "Other"

    def extract_layers(self) -> list[CADLayer]:
        """Extract all layers from the drawing"""
        self.layers = []

        try:
            for layer in self.doc.layers:
                cad_layer = CADLayer(
                    name=layer.dxf.name,
                    color=layer.dxf.color,
                    linetype=layer.dxf.linetype,
                    is_on=layer.is_on(),
                    is_frozen=layer.is_frozen(),
                    is_locked=layer.is_locked(),
                    entity_count=0,  # Will be updated when extracting entities
                )
                self.layers.append(cad_layer)
        except Exception:
            pass

        return self.layers

    def extract_blocks(self) -> list[CADBlock]:
        """Extract block definitions from the drawing"""
        self.blocks = []
        block_insert_counts = defaultdict(int)

        try:
            # Count block insertions
            msp = self.doc.modelspace()
            for entity in msp:
                if entity.dxftype() == "INSERT":
                    block_insert_counts[entity.dxf.name] += 1

            # Extract block definitions
            for block in self.doc.blocks:
                if block.name.startswith("*"):  # Skip anonymous blocks
                    continue

                attrs = []
                for entity in block:
                    if entity.dxftype() == "ATTDEF":
                        attrs.append(
                            {
                                "tag": entity.dxf.tag,
                                "prompt": entity.dxf.prompt if hasattr(entity.dxf, "prompt") else "",
                                "default": entity.dxf.default if hasattr(entity.dxf, "default") else "",
                            }
                        )

                cad_block = CADBlock(
                    name=block.name,
                    base_point=tuple(block.base_point) if block.base_point else (0, 0, 0),
                    entity_count=len(list(block)),
                    insert_count=block_insert_counts.get(block.name, 0),
                    attributes=attrs,
                )
                self.blocks.append(cad_block)
        except Exception:
            pass

        return self.blocks

    def _extract_geometry(self, entity) -> dict:
        """Extract geometry data from an entity"""
        geometry = {}
        etype = entity.dxftype()

        try:
            if etype == "LINE":
                geometry = {
                    "type": "line",
                    "start": {"x": entity.dxf.start.x, "y": entity.dxf.start.y, "z": entity.dxf.start.z},
                    "end": {"x": entity.dxf.end.x, "y": entity.dxf.end.y, "z": entity.dxf.end.z},
                }
            elif etype == "CIRCLE":
                geometry = {
                    "type": "circle",
                    "center": {"x": entity.dxf.center.x, "y": entity.dxf.center.y, "z": entity.dxf.center.z},
                    "radius": entity.dxf.radius,
                }
            elif etype == "ARC":
                geometry = {
                    "type": "arc",
                    "center": {"x": entity.dxf.center.x, "y": entity.dxf.center.y, "z": entity.dxf.center.z},
                    "radius": entity.dxf.radius,
                    "start_angle": entity.dxf.start_angle,
                    "end_angle": entity.dxf.end_angle,
                }
            elif etype == "POINT":
                geometry = {
                    "type": "point",
                    "location": {"x": entity.dxf.location.x, "y": entity.dxf.location.y, "z": entity.dxf.location.z},
                }
            elif etype in ("POLYLINE", "LWPOLYLINE"):
                points = []
                if etype == "LWPOLYLINE":
                    for x, y, _, _, _ in entity.get_points():
                        points.append({"x": x, "y": y, "z": 0})
                else:
                    for v in entity.vertices:
                        points.append({"x": v.dxf.location.x, "y": v.dxf.location.y, "z": v.dxf.location.z})
                geometry = {
                    "type": "polyline",
                    "closed": entity.closed if hasattr(entity, "closed") else False,
                    "points": points,
                }
            elif etype == "SPLINE":
                ctrl_points = []
                for cp in entity.control_points:
                    ctrl_points.append({"x": cp.x, "y": cp.y, "z": cp.z})
                geometry = {
                    "type": "spline",
                    "degree": entity.dxf.degree,
                    "control_points": ctrl_points,
                }
            elif etype == "ELLIPSE":
                geometry = {
                    "type": "ellipse",
                    "center": {"x": entity.dxf.center.x, "y": entity.dxf.center.y, "z": entity.dxf.center.z},
                    "major_axis": {"x": entity.dxf.major_axis.x, "y": entity.dxf.major_axis.y, "z": entity.dxf.major_axis.z},
                    "ratio": entity.dxf.ratio,
                }
            elif etype == "HATCH":
                geometry = {
                    "type": "hatch",
                    "pattern_name": entity.dxf.pattern_name,
                    "pattern_type": entity.dxf.pattern_type,
                }
            elif etype == "INSERT":
                geometry = {
                    "type": "block_reference",
                    "block_name": entity.dxf.name,
                    "insertion_point": {
                        "x": entity.dxf.insert.x,
                        "y": entity.dxf.insert.y,
                        "z": entity.dxf.insert.z,
                    },
                    "scale": {
                        "x": entity.dxf.xscale,
                        "y": entity.dxf.yscale,
                        "z": entity.dxf.zscale,
                    },
                    "rotation": entity.dxf.rotation,
                }
            elif etype in ("TEXT", "MTEXT"):
                geometry = {"type": "text"}
            elif etype == "DIMENSION":
                geometry = {
                    "type": "dimension",
                    "dimension_type": entity.dimtype if hasattr(entity, "dimtype") else "unknown",
                }
            elif etype == "LEADER":
                geometry = {"type": "leader"}
            elif etype == "SOLID":
                geometry = {"type": "solid"}
            elif etype == "3DFACE":
                geometry = {"type": "3dface"}
            elif etype == "MESH":
                geometry = {"type": "mesh"}
            else:
                geometry = {"type": etype.lower()}
        except Exception:
            geometry = {"type": etype.lower(), "error": "Could not extract geometry"}

        return geometry

    def extract_entities(self, include_text_content: bool = True) -> list[CADEntity]:
        """
        Extract all entities from the drawing.

        Args:
            include_text_content: Whether to include text content from TEXT/MTEXT entities

        Returns:
            List of CADEntity objects
        """
        self.entities = []
        layer_counts = defaultdict(int)

        try:
            msp = self.doc.modelspace()
            for entity in msp:
                try:
                    etype = entity.dxftype()
                    layer_name = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
                    layer_counts[layer_name] += 1

                    cad_entity = CADEntity(
                        handle=entity.dxf.handle if hasattr(entity.dxf, "handle") else "",
                        entity_type=etype,
                        layer=layer_name,
                        color=entity.dxf.color if hasattr(entity.dxf, "color") else 256,
                        linetype=entity.dxf.linetype if hasattr(entity.dxf, "linetype") else "BYLAYER",
                        lineweight=entity.dxf.lineweight if hasattr(entity.dxf, "lineweight") else -1,
                        geometry=self._extract_geometry(entity),
                    )

                    # Extract text content
                    if include_text_content:
                        if etype == "TEXT":
                            cad_entity.text_content = entity.dxf.text
                        elif etype == "MTEXT":
                            cad_entity.text_content = entity.text if hasattr(entity, "text") else ""

                    # Extract block attributes
                    if etype == "INSERT":
                        cad_entity.block_name = entity.dxf.name
                        try:
                            for attrib in entity.attribs:
                                cad_entity.attributes[attrib.dxf.tag] = attrib.dxf.text
                        except Exception:
                            pass

                    self.entities.append(cad_entity)
                except Exception:
                    continue
        except Exception:
            pass

        # Update layer entity counts
        for layer in self.layers:
            layer.entity_count = layer_counts.get(layer.name, 0)

        return self.entities

    def detect_issues(self) -> list[CADIssue]:
        """
        Detect common issues in the CAD file.

        Returns:
            List of CADIssue objects describing detected issues
        """
        if not self.entities:
            self.extract_entities()
        if not self.layers:
            self.extract_layers()

        self.issues = []

        # Track entities by location for duplicate detection
        entity_locations = defaultdict(list)

        for entity in self.entities:
            # Check for entities on layer 0 (often indicates poor layer management)
            if entity.layer == "0" and entity.entity_type not in ("VIEWPORT", "DIMENSION"):
                self.issues.append(
                    CADIssue(
                        issue_type="entity_on_layer_0",
                        severity="warning",
                        entity_handle=entity.handle,
                        entity_type=entity.entity_type,
                        layer=entity.layer,
                        description=f"{entity.entity_type} entity is on layer 0",
                        suggestion="Move entity to an appropriate named layer for better organization",
                    )
                )

            # Check for zero-length lines
            if entity.entity_type == "LINE" and entity.geometry.get("type") == "line":
                start = entity.geometry.get("start", {})
                end = entity.geometry.get("end", {})
                length = math.sqrt(
                    (end.get("x", 0) - start.get("x", 0)) ** 2
                    + (end.get("y", 0) - start.get("y", 0)) ** 2
                    + (end.get("z", 0) - start.get("z", 0)) ** 2
                )
                if length < 0.0001:  # Effectively zero length
                    self.issues.append(
                        CADIssue(
                            issue_type="zero_length_line",
                            severity="minor",
                            entity_handle=entity.handle,
                            entity_type=entity.entity_type,
                            layer=entity.layer,
                            description="Line has zero or near-zero length",
                            suggestion="Delete zero-length line or verify if it's intentional",
                            location=(start.get("x", 0), start.get("y", 0), start.get("z", 0)),
                        )
                    )

            # Check for zero-radius circles
            if entity.entity_type == "CIRCLE" and entity.geometry.get("type") == "circle":
                if entity.geometry.get("radius", 0) < 0.0001:
                    center = entity.geometry.get("center", {})
                    self.issues.append(
                        CADIssue(
                            issue_type="zero_radius_circle",
                            severity="minor",
                            entity_handle=entity.handle,
                            entity_type=entity.entity_type,
                            layer=entity.layer,
                            description="Circle has zero or near-zero radius",
                            suggestion="Delete zero-radius circle or verify if it's a point",
                            location=(center.get("x", 0), center.get("y", 0), center.get("z", 0)),
                        )
                    )

            # Track entity locations for duplicate detection
            if entity.geometry.get("type") == "line":
                start = entity.geometry.get("start", {})
                end = entity.geometry.get("end", {})
                loc_key = (
                    round(start.get("x", 0), 3),
                    round(start.get("y", 0), 3),
                    round(end.get("x", 0), 3),
                    round(end.get("y", 0), 3),
                    entity.entity_type,
                    entity.layer,
                )
                entity_locations[loc_key].append(entity)

            # Check for empty text
            if entity.entity_type in ("TEXT", "MTEXT") and not entity.text_content.strip():
                self.issues.append(
                    CADIssue(
                        issue_type="empty_text",
                        severity="warning",
                        entity_handle=entity.handle,
                        entity_type=entity.entity_type,
                        layer=entity.layer,
                        description="Text entity has no content",
                        suggestion="Delete empty text or add content",
                    )
                )

        # Check for duplicate entities
        for loc_key, entities_at_loc in entity_locations.items():
            if len(entities_at_loc) > 1:
                for ent in entities_at_loc[1:]:  # Skip first, report duplicates
                    self.issues.append(
                        CADIssue(
                            issue_type="duplicate_entity",
                            severity="major",
                            entity_handle=ent.handle,
                            entity_type=ent.entity_type,
                            layer=ent.layer,
                            description=f"Duplicate {ent.entity_type} found at same location on same layer",
                            suggestion="Delete duplicate entity to reduce file size and avoid confusion",
                            location=(loc_key[0], loc_key[1], 0),
                        )
                    )

        # Check for layers with no entities
        for layer in self.layers:
            if layer.entity_count == 0 and not layer.name.startswith("*"):
                self.issues.append(
                    CADIssue(
                        issue_type="empty_layer",
                        severity="minor",
                        entity_handle="",
                        entity_type="LAYER",
                        layer=layer.name,
                        description=f"Layer '{layer.name}' has no entities",
                        suggestion="Delete unused layer or verify if it's reserved for future use",
                    )
                )

            # Check for frozen or off layers with content (might be intentional but worth noting)
            if layer.entity_count > 0 and (layer.is_frozen or not layer.is_on):
                status = "frozen" if layer.is_frozen else "turned off"
                self.issues.append(
                    CADIssue(
                        issue_type="hidden_layer_content",
                        severity="warning",
                        entity_handle="",
                        entity_type="LAYER",
                        layer=layer.name,
                        description=f"Layer '{layer.name}' is {status} but contains {layer.entity_count} entities",
                        suggestion="Verify if hidden content is intentional or should be visible/deleted",
                    )
                )

        return self.issues

    def to_json(
        self,
        output_path: Optional[str] = None,
        include_entities: bool = True,
        include_text_content: bool = True,
        detect_issues_flag: bool = True,
    ) -> dict:
        """
        Convert CAD drawing to LLM-friendly JSON format.

        Args:
            output_path: Optional path to write JSON file
            include_entities: Include all entity data
            include_text_content: Include text content from TEXT/MTEXT
            detect_issues_flag: Run issue detection

        Returns:
            Dictionary containing the full drawing analysis
        """
        # Extract data if not already done
        if not self.layers:
            self.extract_layers()
        if not self.blocks:
            self.extract_blocks()
        if not self.entities:
            self.extract_entities(include_text_content=include_text_content)

        # Detect issues if requested
        if detect_issues_flag:
            self.detect_issues()

        # Build the JSON structure
        result = {
            "format_version": "1.0",
            "format_type": "cad_analysis",
            "description": "CAD file analysis for LLM processing - includes entities, layers, blocks, and issues",
            "drawing": self.get_drawing_info(),
            "statistics": {
                "total_entities": len(self.entities),
                "entities_by_type": self._get_entity_summary(),
                "entities_by_layer": self._count_by_layer(),
                "entities_by_discipline": self._count_by_discipline(),
                "total_layers": len(self.layers),
                "total_blocks": len(self.blocks),
                "total_issues": len(self.issues),
                "issues_by_severity": self._count_issues_by_severity(),
            },
            "layers": [l.to_dict() for l in self.layers],
            "blocks": [b.to_dict() for b in self.blocks],
            "issues": [i.to_dict() for i in self.issues],
            "text_content": self._extract_all_text(),
            "llm_analysis_hints": {
                "layer_analysis": [
                    "Check layer naming conventions (AIA, ISO, or custom standards)",
                    "Identify discipline based on layer prefixes (A- Architecture, S- Structure, etc.)",
                    "Look for inconsistent layer usage or naming",
                ],
                "drawing_quality": [
                    "Check for duplicate entities at same locations",
                    "Verify proper layer organization",
                    "Look for entities on layer 0 that should be elsewhere",
                ],
                "common_patterns": [
                    "Title blocks are usually INSERT entities on specific layers",
                    "Dimensions should be on dedicated dimension layers",
                    "Text entities often contain important annotations",
                ],
            },
        }

        # Add entities only if requested (can be large)
        if include_entities:
            result["entities"] = [e.to_dict() for e in self.entities]

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    def _count_by_layer(self) -> dict:
        """Count entities by layer"""
        counts = defaultdict(int)
        for entity in self.entities:
            counts[entity.layer] += 1
        return dict(counts)

    def _count_by_discipline(self) -> dict:
        """Count entities by discipline"""
        counts = defaultdict(int)
        for entity in self.entities:
            discipline = self._get_discipline(entity.layer)
            counts[discipline] += 1
        return dict(counts)

    def _count_issues_by_severity(self) -> dict:
        """Count issues by severity"""
        counts = defaultdict(int)
        for issue in self.issues:
            counts[issue.severity] += 1
        return dict(counts)

    def _extract_all_text(self) -> list[dict]:
        """Extract all text content from the drawing"""
        texts = []
        for entity in self.entities:
            if entity.text_content:
                texts.append(
                    {
                        "layer": entity.layer,
                        "type": entity.entity_type,
                        "content": entity.text_content,
                    }
                )
        return texts
