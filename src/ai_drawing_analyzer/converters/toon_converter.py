"""Toon format converter using Node.js bridge"""
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from ..utils.logging import logger


class ToonConverter:
    """Convert JSONL output to Toon format using Node.js"""

    def __init__(self):
        """Initialize Toon converter"""
        self.script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "convert_to_toon.mjs"
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if Node.js and required packages are available"""
        # Check for node
        if not shutil.which("node"):
            raise RuntimeError(
                "Node.js is not installed. Please install Node.js to use Toon format.\n"
                "Visit: https://nodejs.org/"
            )

        # Check if script exists
        if not self.script_path.exists():
            raise FileNotFoundError(
                f"Toon converter script not found at: {self.script_path}\n"
                "Run: pnpm install @toon-format/toon"
            )

        return True

    def convert(self, input_jsonl: str, output_toon: Optional[str] = None) -> str:
        """
        Convert JSONL file to Toon format.

        Args:
            input_jsonl: Path to input JSONL file
            output_toon: Path to output Toon file (default: input_name.toon)

        Returns:
            Path to output Toon file

        Raises:
            RuntimeError: If Node.js is not available
            FileNotFoundError: If input file doesn't exist
            subprocess.CalledProcessError: If conversion fails
        """
        input_path = Path(input_jsonl)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_jsonl}")

        # Default output path
        if output_toon is None:
            output_toon = str(input_path.with_suffix('.toon'))

        logger.info(f"Converting to Toon format: {input_jsonl} -> {output_toon}")

        try:
            # Run Node.js conversion script
            result = subprocess.run(
                ["node", str(self.script_path), str(input_jsonl), output_toon],
                capture_output=True,
                text=True,
                check=True,
                timeout=60
            )

            # Log Node.js output
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")

            logger.info(f"✅ Toon conversion complete: {output_toon}")
            return output_toon

        except subprocess.TimeoutExpired:
            raise RuntimeError("Toon conversion timed out (60s)")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"Toon conversion failed: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Toon conversion: {e}")

    def is_available(self) -> bool:
        """Check if Toon converter is available"""
        try:
            self._check_dependencies()
            return True
        except (RuntimeError, FileNotFoundError):
            return False
