"""
AI Drawing Analyzer - Hugging Face Spaces Entry Point

This file serves as the entry point for Hugging Face Spaces deployment.
It imports and launches the Gradio interface from the main package.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_drawing_analyzer.gui import create_interface
from ai_drawing_analyzer.utils.config import load_env_file

# Load environment variables (API keys from HF Secrets)
load_env_file()

# Create and launch the interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch()
else:
    # For HF Spaces - expose the demo object
    demo.launch()
