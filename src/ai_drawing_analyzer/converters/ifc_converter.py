"""IFC to JSON/Toon converter for LLM analysis"""

import json
from pathlib import Path
from typing import Optional
from ..processing.ifc import IFCProcessor
from ..utils.logging import logger


class IFCConverter:
    """Convert IFC files to JSON format suitable for LLM analysis"""

    def __init__(self):
        """Initialize IFC converter"""
        pass

    def convert_to_json(
        self,
        input_ifc: str,
        output_json: Optional[str] = None,
        include_geometry: bool = True,
        detect_clashes: bool = True,
        detect_issues: bool = True,
    ) -> str:
        """
        Convert IFC file to JSON format.

        Args:
            input_ifc: Path to input IFC file
            output_json: Path to output JSON file (default: input_name.json)
            include_geometry: Include bounding box information
            detect_clashes: Run clash detection
            detect_issues: Run issue detection

        Returns:
            Path to output JSON file

        Raises:
            FileNotFoundError: If input file doesn't exist
            ImportError: If ifcopenshell is not installed
        """
        input_path = Path(input_ifc)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_ifc}")

        # Default output path
        if output_json is None:
            output_json = str(input_path.with_suffix(".json"))

        logger.info(f"Converting IFC to JSON: {input_ifc} -> {output_json}")

        with IFCProcessor(str(input_path)) as processor:
            result = processor.to_json(
                output_path=output_json,
                include_geometry=include_geometry,
                detect_clashes_flag=detect_clashes,
                detect_issues_flag=detect_issues,
            )

            # Log summary
            stats = result.get("statistics", {})
            logger.info(f"  Elements: {stats.get('total_elements', 0)}")
            logger.info(f"  Clashes: {stats.get('total_clashes', 0)}")
            logger.info(f"  Issues: {stats.get('total_issues', 0)}")

        logger.info(f"✅ IFC to JSON conversion complete: {output_json}")
        return output_json

    def convert_to_toon(
        self,
        input_ifc: str,
        output_toon: Optional[str] = None,
        include_geometry: bool = True,
        detect_clashes: bool = True,
        detect_issues: bool = True,
    ) -> str:
        """
        Convert IFC file to Toon format via JSON.

        Args:
            input_ifc: Path to input IFC file
            output_toon: Path to output Toon file (default: input_name.toon)
            include_geometry: Include bounding box information
            detect_clashes: Run clash detection
            detect_issues: Run issue detection

        Returns:
            Path to output Toon file

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If Node.js or Toon package not available
        """
        from .toon_converter import ToonConverter

        input_path = Path(input_ifc)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_ifc}")

        # First convert to JSONL format (required by ToonConverter)
        temp_jsonl = str(input_path.with_suffix(".ifc.jsonl"))

        logger.info(f"Converting IFC to Toon: {input_ifc}")

        # Convert IFC to JSON data
        with IFCProcessor(str(input_path)) as processor:
            result = processor.to_json(
                output_path=None,
                include_geometry=include_geometry,
                detect_clashes_flag=detect_clashes,
                detect_issues_flag=detect_issues,
            )

        # Write as JSONL (one JSON object per line format expected by Toon)
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            # Write metadata
            f.write(
                json.dumps(
                    {
                        "page": 0,
                        "page_type": "ifc_metadata",
                        "provider": "ifc_processor",
                        "model": "ifcopenshell",
                        "text_content": json.dumps(
                            {
                                "format_version": result["format_version"],
                                "format_type": result["format_type"],
                                "project": result["project"],
                                "statistics": result["statistics"],
                            }
                        ),
                    }
                )
                + "\n"
            )

            # Write clashes as a page
            if result.get("clashes"):
                f.write(
                    json.dumps(
                        {
                            "page": 1,
                            "page_type": "ifc_clashes",
                            "provider": "ifc_processor",
                            "model": "ifcopenshell",
                            "text_content": json.dumps(
                                {
                                    "clashes": result["clashes"],
                                    "clash_summary": result["statistics"].get(
                                        "clashes_by_severity", {}
                                    ),
                                }
                            ),
                        }
                    )
                    + "\n"
                )

            # Write issues as a page
            if result.get("issues"):
                f.write(
                    json.dumps(
                        {
                            "page": 2,
                            "page_type": "ifc_issues",
                            "provider": "ifc_processor",
                            "model": "ifcopenshell",
                            "text_content": json.dumps(
                                {
                                    "issues": result["issues"],
                                    "issue_summary": result["statistics"].get(
                                        "issues_by_severity", {}
                                    ),
                                }
                            ),
                        }
                    )
                    + "\n"
                )

            # Write elements in batches
            batch_size = 100
            elements = result.get("elements", [])
            for i in range(0, len(elements), batch_size):
                batch = elements[i : i + batch_size]
                f.write(
                    json.dumps(
                        {
                            "page": 3 + i // batch_size,
                            "page_type": "ifc_elements",
                            "provider": "ifc_processor",
                            "model": "ifcopenshell",
                            "text_content": json.dumps(
                                {"elements_batch": i // batch_size, "elements": batch}
                            ),
                        }
                    )
                    + "\n"
                )

            # Write LLM hints
            f.write(
                json.dumps(
                    {
                        "page": 3 + (len(elements) // batch_size) + 1,
                        "page_type": "ifc_llm_hints",
                        "provider": "ifc_processor",
                        "model": "ifcopenshell",
                        "text_content": json.dumps(result.get("llm_analysis_hints", {})),
                    }
                )
                + "\n"
            )

        # Convert JSONL to Toon
        toon_converter = ToonConverter()
        if output_toon is None:
            output_toon = str(input_path.with_suffix(".toon"))

        result_path = toon_converter.convert(temp_jsonl, output_toon)

        # Clean up temp file
        try:
            Path(temp_jsonl).unlink()
        except Exception:
            pass

        logger.info(f"✅ IFC to Toon conversion complete: {result_path}")
        return result_path

    def analyze_for_llm(
        self,
        input_ifc: str,
        include_elements: bool = False,
    ) -> dict:
        """
        Analyze IFC file and return structured data optimized for LLM consumption.

        This method returns a summary focused on clashes and issues,
        which is what LLMs typically need for analysis.

        Args:
            input_ifc: Path to input IFC file
            include_elements: Whether to include all element details

        Returns:
            Dictionary with analysis results focused on actionable items
        """
        input_path = Path(input_ifc)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_ifc}")

        with IFCProcessor(str(input_path)) as processor:
            full_result = processor.to_json(
                output_path=None,
                include_geometry=True,
                detect_clashes_flag=True,
                detect_issues_flag=True,
            )

        # Build LLM-focused summary
        summary = {
            "project_summary": {
                "name": full_result["project"].get("name", "Unknown"),
                "file": full_result["project"].get("file_name", ""),
                "total_elements": full_result["statistics"]["total_elements"],
                "elements_by_discipline": full_result["statistics"].get(
                    "elements_by_discipline", {}
                ),
            },
            "clash_analysis": {
                "total_clashes": full_result["statistics"]["total_clashes"],
                "by_severity": full_result["statistics"].get("clashes_by_severity", {}),
                "critical_clashes": [
                    c for c in full_result["clashes"] if c["severity"] == "critical"
                ],
                "major_clashes": [
                    c for c in full_result["clashes"] if c["severity"] == "major"
                ],
            },
            "issue_analysis": {
                "total_issues": full_result["statistics"]["total_issues"],
                "by_severity": full_result["statistics"].get("issues_by_severity", {}),
                "critical_issues": [
                    i for i in full_result["issues"] if i["severity"] == "critical"
                ],
                "major_issues": [
                    i for i in full_result["issues"] if i["severity"] == "major"
                ],
            },
            "recommendations": self._generate_recommendations(full_result),
            "llm_hints": full_result.get("llm_analysis_hints", {}),
        }

        if include_elements:
            summary["elements"] = full_result.get("elements", [])

        return summary

    def _generate_recommendations(self, analysis: dict) -> list[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []

        stats = analysis.get("statistics", {})
        clashes_by_severity = stats.get("clashes_by_severity", {})
        issues_by_severity = stats.get("issues_by_severity", {})

        # Critical clash recommendations
        critical_clashes = clashes_by_severity.get("critical", 0)
        if critical_clashes > 0:
            recommendations.append(
                f"URGENT: {critical_clashes} critical clashes detected - "
                "structural/MEP coordination required before construction"
            )

        # Major clash recommendations
        major_clashes = clashes_by_severity.get("major", 0)
        if major_clashes > 10:
            recommendations.append(
                f"HIGH PRIORITY: {major_clashes} major clashes - "
                "schedule coordination meeting with relevant disciplines"
            )

        # Missing level recommendations
        missing_level_issues = [
            i for i in analysis.get("issues", []) if i.get("issue_type") == "missing_level"
        ]
        if len(missing_level_issues) > 5:
            recommendations.append(
                f"MODELING: {len(missing_level_issues)} elements missing level assignment - "
                "assign to building storeys for proper scheduling"
            )

        # Duplicate recommendations
        duplicates = [
            i for i in analysis.get("issues", []) if i.get("issue_type") == "potential_duplicate"
        ]
        if duplicates:
            recommendations.append(
                f"CLEANUP: {len(duplicates)} potential duplicate elements detected - "
                "review and remove if unintended"
            )

        if not recommendations:
            recommendations.append("No critical issues detected - model appears well-coordinated")

        return recommendations

    def is_available(self) -> bool:
        """Check if IFC processing is available"""
        try:
            from ..processing.ifc import HAS_IFC

            return HAS_IFC
        except ImportError:
            return False
