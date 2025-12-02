"""Processing modules for various file formats"""

from .pdf import PDFProcessor

# IFC processor (optional - requires ifcopenshell)
try:
    from .ifc import IFCProcessor, IFCElement, BoundingBox, ClashInfo, IssueInfo
except ImportError:
    IFCProcessor = None
    IFCElement = None
    BoundingBox = None
    ClashInfo = None
    IssueInfo = None

# DWG/DXF processor (optional - requires ezdxf)
try:
    from .dwg import DWGProcessor, CADEntity, CADLayer, CADBlock, CADIssue
except ImportError:
    DWGProcessor = None
    CADEntity = None
    CADLayer = None
    CADBlock = None
    CADIssue = None

__all__ = [
    "PDFProcessor",
    "IFCProcessor",
    "IFCElement",
    "BoundingBox",
    "ClashInfo",
    "IssueInfo",
    "DWGProcessor",
    "CADEntity",
    "CADLayer",
    "CADBlock",
    "CADIssue",
]
