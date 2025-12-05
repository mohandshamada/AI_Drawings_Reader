#!/bin/bash
#
# AI Drawing Analyzer - One-Time Installation Script
# This script installs all dependencies required to run the tool
#

set -e  # Exit on error

echo "🚀 AI Drawing Analyzer - Installation Script"
echo "=============================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

# Require Python 3.9+
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "[ERROR] Python 3.9 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "[OK] Python $PYTHON_VERSION detected"
echo ""

# Installation options
echo "📋 Installation Options:"
echo ""
echo "1. Minimal - Core dependencies only (cloud APIs)"
echo "2. Local  - With local model support (requires ~10GB disk space)"
echo "3. Full   - Everything including development tools"
echo "4. Dev    - Development only (testing, linting, formatting)"
echo ""
read -p "Select installation type (1-4): " INSTALL_TYPE
echo ""

case $INSTALL_TYPE in
    1)
        INSTALL_EXTRAS=""
        INSTALL_NAME="Minimal"
        ;;
    2)
        INSTALL_EXTRAS=".[local]"
        INSTALL_NAME="Local"
        ;;
    3)
        INSTALL_EXTRAS=".[full,dev]"
        INSTALL_NAME="Full"
        ;;
    4)
        INSTALL_EXTRAS=".[dev]"
        INSTALL_NAME="Development"
        ;;
    *)
        echo "❌ Invalid selection"
        exit 1
        ;;
esac

echo "📦 Installing: $INSTALL_NAME"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ pip upgraded"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -n "$INSTALL_EXTRAS" ]; then
    pip install -e "$INSTALL_EXTRAS"
else
    pip install -e .
fi
echo "✅ Python dependencies installed"
echo ""

# Install Node.js dependencies for Toon format
read -p "📦 Install Node.js dependencies for Toon format export? (y/n): " INSTALL_NODE

if [[ "$INSTALL_NODE" =~ ^[Yy]$ ]]; then
    if command -v npm &> /dev/null; then
        echo "🔧 Installing Node.js dependencies..."
        npm install
        echo "✅ Node.js dependencies installed"
    elif command -v pnpm &> /dev/null; then
        echo "🔧 Installing Node.js dependencies with pnpm..."
        pnpm install
        echo "✅ Node.js dependencies installed"
    else
        echo "⚠️  Node.js/npm not found. Skipping Toon format support."
        echo "   Install Node.js from: https://nodejs.org/"
    fi
    echo ""
fi

# GPU Setup for local models
if [[ "$INSTALL_TYPE" == "2" ]] || [[ "$INSTALL_TYPE" == "3" ]]; then
    echo ""
    read -p "🎮 Do you have an NVIDIA GPU and want GPU acceleration? (y/n): " INSTALL_GPU

    if [[ "$INSTALL_GPU" =~ ^[Yy]$ ]]; then
        echo "🔧 Installing PyTorch with CUDA support..."
        echo "   This will download ~2GB of data..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        echo "✅ GPU support installed"
    fi
    echo ""
fi

# Verify installation
echo "🔍 Verifying installation..."
echo ""

# Check if command is available
if command -v ai-drawing-analyzer &> /dev/null; then
    echo "✅ ai-drawing-analyzer command installed successfully"
else
    echo "❌ Installation failed: ai-drawing-analyzer command not found"
    exit 1
fi

# Test import
python3 -c "import ai_drawing_analyzer; print('✅ Python package imports successfully')" || {
    echo "❌ Installation failed: Cannot import package"
    exit 1
}

echo ""
echo "✨ Installation Complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next Steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set up API keys in .env file:"
echo "   cp .env.example .env  # If .env.example exists"
echo "   nano .env"
echo ""
echo "3. Run the tool:"
echo "   ai-drawing-analyzer your_document.pdf"
echo ""
echo "4. For help:"
echo "   ai-drawing-analyzer --help"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Save activation hint
echo ""
echo "💡 Tip: To use the tool in future sessions, run:"
echo "   source venv/bin/activate"
echo ""
