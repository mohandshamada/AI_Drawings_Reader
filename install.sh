#!/bin/bash
# Installation script for PDF Analyzer CLI (Linux/Mac)
# Supports Florence-2 and any Hugging Face vision model
# Uses uv for dependency management

echo "🚀 AI Drawing Analyzer - Installation (Linux/macOS)"
echo "======================================================"
echo ""
echo "This script will help you set up the PDF Analyzer for:"
echo "  ✅ Florence-2 OCR (local, no API key)"
echo "  ✅ Any Hugging Face vision model locally"
echo "  ✅ Cloud APIs (Google Gemini, OpenAI, Anthropic, etc.)"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Found Python $PYTHON_VERSION"

# Check and install uv if needed
if ! command -v uv &> /dev/null; then
    echo ""
    echo "📦 Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install uv. Please install manually from https://astral.sh/uv"
        exit 1
    fi
    echo "✅ uv installed successfully"
else
    echo "✅ Found uv ($(uv --version))"
fi

# Ask user for installation type
echo ""
echo "Choose your installation type:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  LOCAL MODELS ONLY (Recommended)"
echo "   • Run Florence-2, Qwen-VL, Qwen3-VL, LLaVA locally"
echo "   • No API key required"
echo "   • Best for: Technical drawings, blueprints"
echo ""
echo "2️⃣  CLOUD APIs ONLY (Minimal)"
echo "   • Use Gemini, OpenAI, Claude, etc."
echo "   • Requires API keys"
echo "   • Best for: Quick testing, occasional use"
echo ""
echo "3️⃣  BOTH LOCAL & CLOUD (Full Setup)"
echo "   • All features: local models + cloud APIs"
echo "   • Choose which to use at runtime"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "Enter choice (1-3) [default: 1]: " choice
choice=${choice:-1}

# Install dependencies based on choice
echo ""
echo "📥 Installing dependencies with uv..."
echo "   (This may take 2-5 minutes depending on your internet)"
echo ""

case $choice in
    1)
        echo "📚 Installing LOCAL MODEL SUPPORT..."
        echo "   • transformers, torch, timm, einops"
        uv sync --extra local
        echo ""
        echo "🚀 GPU ACCELERATION (Optional but Recommended)"
        echo "   To use GPU (CUDA 11.8) for 10-100x faster inference:"
        echo ""
        echo "   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        echo ""
        echo "   For other CUDA versions, visit: https://pytorch.org/get-started/locally/"
        ;;
    2)
        echo "☁️  Installing CLOUD API SUPPORT..."
        echo "   • httpx, pymupdf, Pillow, google-auth, etc."
        uv sync
        echo ""
        ;;
    3)
        echo "🔗 Installing FULL SETUP (Local + Cloud)..."
        echo "   • All local model dependencies"
        echo "   • All cloud API dependencies"
        uv sync --all-extras
        echo ""
        echo "🚀 GPU ACCELERATION (Optional but Recommended)"
        echo "   To use GPU (CUDA 11.8) for faster inference:"
        echo ""
        echo "   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        echo ""
        ;;
    *)
        echo "❓ Invalid choice. Using LOCAL MODELS (option 1) by default..."
        uv sync --extra local
        ;;
esac

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo ""
        echo "📝 Creating .env file from template..."
        cp .env.example .env
        if [ "$choice" = "1" ]; then
            echo "✅ .env file created (not needed for local models)"
        else
            echo "✅ .env file created. Configure API keys to use cloud providers."
        fi
    fi
fi

echo ""
echo "======================================================"
echo "✅ Installation Complete!"
echo "======================================================"
echo ""
echo "🎯 NEXT STEPS:"
echo ""

if [ "$choice" = "1" ]; then
    echo "1️⃣  Activate environment:"
    echo "    source .venv/bin/activate"
    echo ""
    echo "2️⃣  Test with Florence-2 (interactive):"
    echo "    ai-drawing-analyzer your_document.pdf"
    echo ""
    echo "3️⃣  Or use command-line directly:"
    echo "    ai-drawing-analyzer doc.pdf -p huggingface-local -m microsoft/Florence-2-large"
    echo ""
    echo "📊 First run will download the model (~2-24GB, takes 5-10 minutes)"
    echo "   Model is cached afterwards for fast reuse"
elif [ "$choice" = "2" ]; then
    echo "1️⃣  Activate environment:"
    echo "    source .venv/bin/activate"
    echo ""
    echo "2️⃣  Add API keys to .env file:"
    echo "    nano .env"
    echo ""
    echo "3️⃣  Test with Gemini (free tier):"
    echo "    ai-drawing-analyzer your_document.pdf -p gemini"
    echo ""
    echo "🆓 Get free API keys:"
    echo "   • Google Gemini: https://makersuite.google.com/app/apikey"
    echo "   • HuggingFace Router: https://huggingface.co/settings/tokens"
else
    echo "1️⃣  Activate environment:"
    echo "    source .venv/bin/activate"
    echo ""
    echo "2️⃣  Choose your path:"
    echo ""
    echo "   LOCAL (no API key):"
    echo "   ai-drawing-analyzer doc.pdf -p huggingface-local"
    echo ""
    echo "   CLOUD (with API key):"
    echo "   nano .env  (add your API keys)"
    echo "   ai-drawing-analyzer doc.pdf -p gemini"
    echo ""
fi

echo ""
echo "📖 Documentation:"
echo "   • Quick Start: QUICK_START.md"
echo "   • Full Guide: README.md"
echo "   • Help: ai-drawing-analyzer --help"
echo ""
echo "======================================================"
echo ""
