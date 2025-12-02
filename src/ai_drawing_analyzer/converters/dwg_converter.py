"""DWG/DXF to JSON/Toon converter for LLM analysis"""

import json
from pathlib import Path
from typing import Optional
from ..processing.dwg import DWGProcessor
from ..utils.logging import logger


class DWGConverter:
    """Convert DWG/DXF files to JSON format suitable for LLM analysis"""

    def __init__(self, oda_converter_path: Optional[str] = None):
        """
        Initialize DWG converter.

        Args:
            oda_converter_path: Path to ODA File Converter for DWG support
        """
        self.oda_converter_path = oda_converter_path

    def convert_to_json(
        self,
        input_cad: str,
        output_json: Optional[str] = None,
        include_entities: bool = True,
        include_text_content: bool = True,
        detect_issues: bool = True,
    ) -> str:
        """
        Convert DWG/DXF file to JSON format.

        Args:
            input_cad: Path to input DWG or DXF file
            output_json: Path to output JSON file (default: input_name.json)
            include_entities: Include all entity data
            include_text_content: Include text content from TEXT/MTEXT
            detect_issues: Run issue detection

        Returns:
            Path to output JSON file

        Raises:
            FileNotFoundError: If input file doesn't exist
            ImportError: If ezdxf is not installed
        """
        input_path = Path(input_cad)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_cad}")

        # Default output path
        if output_json is None:
            output_json = str(input_path.with_suffix(".json"))

        logger.info(f"Converting CAD to JSON: {input_cad} -> {output_json}")

        with DWGProcessor(str(input_path), self.oda_converter_path) as processor:
            result = processor.to_json(
                output_path=output_json,
                include_entities=include_entities,
                include_text_content=include_text_content,
                detect_issues_flag=detect_issues,
            )

            # Log summary
            stats = result.get("statistics", {})
            logger.info(f"  Entities: {stats.get('total_entities', 0)}")
            logger.info(f"  Layers: {stats.get('total_layers', 0)}")
            logger.info(f"  Issues: {stats.get('total_issues', 0)}")

        logger.info(f"✅ CAD to JSON conversion complete: {output_json}")
        return output_json

    def convert_to_toon(
        self,
        input_cad: str,
        output_toon: Optional[str] = None,
        include_entities: bool = True,
        include_text_content: bool = True,
        detect_issues: bool = True,
    ) -> str:
        """
        Convert DWG/DXF file to Toon format via JSON.

        Args:
            input_cad: Path to input DWG or DXF file
            output_toon: Path to output Toon file (default: input_name.toon)
            include_entities: Include all entity data
            include_text_content: Include text content
            detect_issues: Run issue detection

        Returns:
            Path to output Toon file

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If Node.js or Toon package not available
        """
        from .toon_converter import ToonConverter

        input_path = Path(input_cad)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_cad}")

        # First convert to JSONL format (required by ToonConverter)
        temp_jsonl = str(input_path.with_suffix(".cad.jsonl"))

        logger.info(f"Converting CAD to Toon: {input_cad}")

        # Convert CAD to JSON data
        with DWGProcessor(str(input_path), self.oda_converter_path) as processor:
            result = processor.to_json(
                output_path=None,
                include_entities=include_entities,
                include_text_content=include_text_content,
                detect_issues_flag=detect_issues,
            )

        # Write as JSONL (one JSON object per line format expected by Toon)
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            # Write metadata
            f.write(
                json.dumps(
                    {
                        "page": 0,
                        "page_type": "cad_metadata",
                        "provider": "cad_processor",
                        "model": "ezdxf",
                        "text_content": json.dumps(
                            {
                                "format_version": result["format_version"],
                                "format_type": result["format_type"],
                                "drawing": result["drawing"],
                                "statistics": result["statistics"],
                            }
                        ),
                    }
                )
                + "\n"
            )

            # Write layers
            f.write(
                json.dumps(
                    {
                        "page": 1,
                        "page_type": "cad_layers",
                        "provider": "cad_processor",
                        "model": "ezdxf",
                        "text_content": json.dumps({"layers": result.get("layers", [])}),
                    }
                )
                + "\n"
            )

            # Write blocks
            f.write(
                json.dumps(
                    {
                        "page": 2,
                        "page_type": "cad_blocks",
                        "provider": "cad_processor",
                        "model": "ezdxf",
                        "text_content": json.dumps({"blocks": result.get("blocks", [])}),
                    }
                )
                + "\n"
            )

            # Write issues
            if result.get("issues"):
                f.write(
                    json.dumps(
                        {
                            "page": 3,
                            "page_type": "cad_issues",
                            "provider": "cad_processor",
                            "model": "ezdxf",
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

            # Write text content as searchable page
            if result.get("text_content"):
                f.write(
                    json.dumps(
                        {
                            "page": 4,
                            "page_type": "cad_text",
                            "provider": "cad_processor",
                            "model": "ezdxf",
                            "text_content": json.dumps(
                                {"text_annotations": result["text_content"]}
                            ),
                        }
                    )
                    + "\n"
                )

            # Write entities in batches
            if include_entities:
                batch_size = 100
                entities = result.get("entities", [])
                for i in range(0, len(entities), batch_size):
                    batch = entities[i : i + batch_size]
                    f.write(
                        json.dumps(
                            {
                                "page": 5 + i // batch_size,
                                "page_type": "cad_entities",
                                "provider": "cad_processor",
                                "model": "ezdxf",
                                "text_content": json.dumps(
                                    {"entities_batch": i // batch_size, "entities": batch}
                                ),
                            }
                        )
                        + "\n"
                    )

            # Write LLM hints
            page_num = 5 + (len(result.get("entities", [])) // batch_size) + 1
            f.write(
                json.dumps(
                    {
                        "page": page_num,
                        "page_type": "cad_llm_hints",
                        "provider": "cad_processor",
                        "model": "ezdxf",
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

        logger.info(f"✅ CAD to Toon conversion complete: {result_path}")
        return result_path

    def analyze_for_llm(
        self,
        input_cad: str,
        include_entities: bool = False,
    ) -> dict:
        """
        Analyze CAD file and return structured data optimized for LLM consumption.

        This method returns a summary focused on drawing structure and issues,
        which is what LLMs typically need for analysis.

        Args:
            input_cad: Path to input DWG/DXF file
            include_entities: Whether to include all entity details

        Returns:
            Dictionary with analysis results focused on actionable items
        """
        input_path = Path(input_cad)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_cad}")

        with DWGProcessor(str(input_path), self.oda_converter_path) as processor:
            full_result = processor.to_json(
                output_path=None,
                include_entities=include_entities,
                include_text_content=True,
                detect_issues_flag=True,
            )

        # Build LLM-focused summary
        summary = {
            "drawing_summary": {
                "file_name": full_result["drawing"].get("file_name", ""),
                "file_format": full_result["drawing"].get("file_format", ""),
                "dxf_version": full_result["drawing"].get("dxf_version", ""),
                "units": full_result["drawing"].get("units", {}),
                "total_entities": full_result["statistics"]["total_entities"],
                "entities_by_type": full_result["statistics"].get("entities_by_type", {}),
                "entities_by_discipline": full_result["statistics"].get(
                    "entities_by_discipline", {}
                ),
            },
            "layer_analysis": {
                "total_layers": full_result["statistics"]["total_layers"],
                "layers": [
                    {
                        "name": l["name"],
                        "entity_count": l["entity_count"],
                        "is_on": l["is_on"],
                        "is_frozen": l["is_frozen"],
                    }
                    for l in full_result.get("layers", [])
                ],
            },
            "block_analysis": {
                "total_blocks": full_result["statistics"]["total_blocks"],
                "blocks": [
                    {
                        "name": b["name"],
                        "insert_count": b["insert_count"],
                        "has_attributes": len(b.get("attributes", [])) > 0,
                    }
                    for b in full_result.get("blocks", [])
                ],
            },
            "issue_analysis": {
                "total_issues": full_result["statistics"]["total_issues"],
                "by_severity": full_result["statistics"].get("issues_by_severity", {}),
                "major_issues": [
                    i for i in full_result.get("issues", []) if i["severity"] == "major"
                ],
                "warnings": [
                    i for i in full_result.get("issues", []) if i["severity"] == "warning"
                ],
            },
            "text_content": full_result.get("text_content", []),
            "recommendations": self._generate_recommendations(full_result),
            "llm_hints": full_result.get("llm_analysis_hints", {}),
        }

        if include_entities:
            summary["entities"] = full_result.get("entities", [])

        return summary

    def _generate_recommendations(self, analysis: dict) -> list[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []

        stats = analysis.get("statistics", {})
        issues_by_severity = stats.get("issues_by_severity", {})

        # Duplicate entity recommendations
        duplicate_issues = [
            i for i in analysis.get("issues", []) if i.get("issue_type") == "duplicate_entity"
        ]
        if len(duplicate_issues) > 10:
            recommendations.append(
                f"CLEANUP: {len(duplicate_issues)} duplicate entities detected - "
                "use OVERKILL or similar command to remove duplicates"
            )

        # Layer 0 recommendations
        layer0_issues = [
            i for i in analysis.get("issues", []) if i.get("issue_type") == "entity_on_layer_0"
        ]
        if len(layer0_issues) > 20:
            recommendations.append(
                f"ORGANIZATION: {len(layer0_issues)} entities on layer 0 - "
                "move to appropriate named layers for better management"
            )

        # Empty layers
        empty_layers = [
            i for i in analysis.get("issues", []) if i.get("issue_type") == "empty_layer"
        ]
        if len(empty_layers) > 10:
            recommendations.append(
                f"CLEANUP: {len(empty_layers)} unused layers - "
                "purge unused layers to reduce file size"
            )

        # Hidden content
        hidden_issues = [
            i for i in analysis.get("issues", []) if i.get("issue_type") == "hidden_layer_content"
        ]
        if hidden_issues:
            recommendations.append(
                f"REVIEW: {len(hidden_issues)} layers with hidden content - "
                "verify if frozen/off content is intentional"
            )

        # Empty text
        empty_text = [
            i for i in analysis.get("issues", []) if i.get("issue_type") == "empty_text"
        ]
        if empty_text:
            recommendations.append(
                f"CLEANUP: {len(empty_text)} empty text entities - "
                "delete or add content"
            )

        if not recommendations:
            recommendations.append("No significant issues detected - drawing appears well-organized")

        return recommendations

    def extract_text_annotations(self, input_cad: str) -> list[dict]:
        """
        Extract all text annotations from a CAD file.

        Useful for analyzing drawing notes, dimensions, and annotations.

        Args:
            input_cad: Path to input DWG/DXF file

        Returns:
            List of text annotations with layer and content
        """
        input_path = Path(input_cad)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_cad}")

        with DWGProcessor(str(input_path), self.oda_converter_path) as processor:
            processor.extract_entities(include_text_content=True)
            return [
                {"layer": e.layer, "type": e.entity_type, "content": e.text_content}
                for e in processor.entities
                if e.text_content
            ]

    def is_available(self) -> bool:
        """Check if CAD processing is available"""
        try:
            from ..processing.dwg import HAS_EZDXF

            return HAS_EZDXF
        except ImportError:
            return False

    def can_read_dwg(self) -> bool:
        """Check if DWG files can be read (requires ODA converter)"""
        try:
            from ..processing.dwg import DWGProcessor

            processor = DWGProcessor.__new__(DWGProcessor)
            return processor._find_oda_converter() is not None
        except Exception:
            return False
