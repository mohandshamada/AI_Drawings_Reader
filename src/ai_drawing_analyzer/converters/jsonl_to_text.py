import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any
from ..utils.logging import logger

class DrawingTextConverter:
    """Convert JSONL OCR output to LLM-friendly text format"""

    def __init__(self, jsonl_file: str):
        self.jsonl_file = jsonl_file
        self.pages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, str] = {}
        self.cross_refs: Dict[str, List[int]] = {}
        self.legends: Dict[str, str] = {}
        self.document_structure: str = ""

        # Regex patterns for extraction
        self.patterns = {
            'page_reference': [
                r'(?:see|refer to|detail|section|drawing)\s+([A-Z0-9\-]+).*?(?:page|pg\.?)s*(\d+)',
                r'(?:page|pg\.?)s*(\d+)s+.*?(?:detail|section|schedule)\s+([A-Z0-9\-]+)',
                r'(?:see|ref(?:er)?\s+to)\s+(?:page|pg\.?)s*(\d+)',
                r'on\s+(?:page|pg\.?)s*(\d+)',
                r'(?:page|pg\.?)s*(\d+)',
            ],
            'title_block': {
                'project': r'(?:project|proj\.mdrawing\s+name)[\s:]*([^\n]+)',
                'title': r'(?:title|drawing\s+title|drawing\s+name)[\s:]*([^\n]+)',
                'discipline': r'(?:discipline)[\s:]*([^\n]+)',
                'sheet': r'(?:sheet|drawing\s+number|sheet\s+number)[\s:]*([A-Z0-9\-\.]+)',
                'revision': r'(?:revision|rev\.?)s*([A-Z0-9\-]+)',
                'date': r'(?:date|rev\.mdrawing\s+date)[\s:]*([0-9\-/]+)',
                'scale': r'(?:scale)[\s:]*([0-9\':" =/\-]+)',
                'drawn_by': r'(?:drawn\s+by|drawn)[\s:]*([^\n]+)',
                'checked_by': r'(?:checked\s+by|checked)[\s:]*([^\n]+)',
                'approved_by': r'(?:approved\s+by|approved)[\s:]*([^\n]+)',
            },
            'legend_keywords': ['legend', 'symbols', 'abbreviations', 'notes', 'key', 'symbol key']
        }

    def load_pages(self) -> List[Dict[str, Any]]:
        """Load all pages from JSONL file"""
        try:
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            page_data = json.loads(line)
                            self.pages.append(page_data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipped malformed JSON entry: {e}")

            logger.info(f"Loaded {len(self.pages)} pages from {self.jsonl_file}")
            return self.pages
        except FileNotFoundError:
            logger.error(f"File not found: {self.jsonl_file}")
            raise

    def extract_metadata(self) -> Dict[str, str]:
        """Extract title block info from first page"""
        if not self.pages:
            return {}

        first_page_text = self.pages[0].get('text_content', '')
        self.metadata = {
            'project': 'UNKNOWN',
            'title': 'UNKNOWN',
            'discipline': 'UNKNOWN',
            'sheet': 'UNKNOWN',
            'revision': 'UNKNOWN',
            'date': 'UNKNOWN',
            'scale': 'UNKNOWN',
            'drawn_by': 'UNKNOWN',
            'checked_by': 'UNKNOWN',
            'approved_by': 'UNKNOWN',
        }

        for field, pattern in self.patterns['title_block'].items():
            match = re.search(pattern, first_page_text, re.IGNORECASE | re.MULTILINE)
            if match:
                self.metadata[field] = match.group(1).strip()

        return self.metadata

    def find_cross_references(self) -> Dict[str, List[int]]:
        """Build cross-reference map from all pages"""
        self.cross_refs = {}

        for page_idx, page in enumerate(self.pages):
            text = page.get('text_content', '')
            page_num = page_idx + 1

            for pattern in self.patterns['page_reference']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if groups:
                        if len(groups) >= 2:
                            detail_name = groups[0].strip() if groups[0] else f"Reference on page {page_num}"
                            page_ref = int(groups[1]) if groups[1].isdigit() else page_num
                        else:
                            page_ref = int(groups[0]) if groups[0] and groups[0].isdigit() else page_num
                            detail_name = f"Reference on page {page_num}"

                        if detail_name not in self.cross_refs:
                            self.cross_refs[detail_name] = []
                        if page_ref not in self.cross_refs[detail_name]:
                            self.cross_refs[detail_name].append(page_ref)

        for key in self.cross_refs:
            self.cross_refs[key].sort()

        return self.cross_refs

    def extract_legends(self) -> Dict[str, str]:
        """Find and extract legend/symbol definitions"""
        self.legends = {}

        for page_idx, page in enumerate(self.pages):
            text = page.get('text_content', '')
            page_num = page_idx + 1

            is_legend_page = any(keyword.lower() in text.lower()
                                for keyword in self.patterns['legend_keywords'])

            if is_legend_page:
                legend_key = f"Page {page_num} - Symbols/Legend"
                self.legends[legend_key] = text

        return self.legends

    def create_document_structure(self) -> str:
        """Create table of contents / structure map"""
        if not self.pages:
            return "No pages available"

        structure = "DOCUMENT STRUCTURE:\n"
        structure += "=" * 80 + "\n"

        for page_idx, page in enumerate(self.pages):
            page_num = page_idx + 1
            text = page.get('text_content', '').strip()
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            summary = lines[0][:60] if lines else "Content not available"
            structure += f"Page {page_num}: {summary}\n"

        structure += "=" * 80 + "\n"
        self.document_structure = structure
        return structure

    def assemble_output(self) -> str:
        """Combine all sections into final text"""
        output_parts = []

        # Header
        output_parts.append("=" * 80)
        output_parts.append("COMPLETE DRAWING SET - TEXT EXTRACTION FOR LLM ANALYSIS")
        output_parts.append("=" * 80)
        output_parts.append("")

        # Metadata Section
        output_parts.append("DOCUMENT METADATA")
        output_parts.append("=" * 80)
        for key, value in self.metadata.items():
            display_key = key.replace('_', ' ').title()
            output_parts.append(f"{display_key}: {value}")
        output_parts.append(f"Total Pages: {len(self.pages)}")
        output_parts.append(f"Extracted Date: {datetime.now().isoformat()}Z")
        output_parts.append("=" * 80)
        output_parts.append("")

        # Cross-Reference Index
        if self.cross_refs:
            output_parts.append("CROSS-REFERENCE INDEX")
            output_parts.append("=" * 80)
            for detail, pages in sorted(self.cross_refs.items()):
                pages_str = ", ".join(str(p) for p in sorted(set(pages)))
                output_parts.append(f"{detail}: pages {pages_str}")
            output_parts.append("=" * 80)
            output_parts.append("")

        # Legend Section
        if self.legends:
            output_parts.append("LEGEND & SYMBOLS REFERENCE")
            output_parts.append("=" * 80)
            for legend_name, legend_content in self.legends.items():
                output_parts.append(f"\n{legend_name}:")
                output_parts.append("-" * 80)
                output_parts.append(legend_content[:500])  # Limit to first 500 chars
                output_parts.append("")
            output_parts.append("=" * 80)
            output_parts.append("")

        # Document Structure
        output_parts.append(self.document_structure)
        output_parts.append("")

        # Complete Drawing Content
        output_parts.append("COMPLETE DRAWING CONTENT")
        output_parts.append("=" * 80)
        output_parts.append("")

        for page_idx, page in enumerate(self.pages):
            page_num = page_idx + 1
            total_pages = len(self.pages)

            output_parts.append("-" * 80)
            output_parts.append(f"PAGE {page_num} of {total_pages}")
            output_parts.append("-" * 80)
            output_parts.append(page.get('text_content', ''))
            output_parts.append("")
            output_parts.append("")

        # Footer
        output_parts.append("=" * 80)
        output_parts.append("END OF DRAWING SET")
        output_parts.append(f"Generated: {datetime.now().isoformat()}Z")
        output_parts.append(f"Total Characters: {sum(len(p) for p in output_parts)}")
        output_parts.append(f"Total Pages: {len(self.pages)}")
        output_parts.append(f"Cross-References Found: {len(self.cross_refs)}")
        output_parts.append("=" * 80)

        return "\n".join(output_parts)

    def convert(self, output_path: str = 'drawing_complete.txt') -> str:
        """Main conversion pipeline"""
        try:
            logger.info("Converting JSONL to text format...")
            self.load_pages()
            self.extract_metadata()
            self.find_cross_references()
            self.extract_legends()
            self.create_document_structure()
            
            content = self.assemble_output()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Saved text conversion to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise
