# AI Drawing Analyzer

**Transform PDFs into structured text using state-of-the-art Vision Language Models**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-13%20passing-brightgreen.svg)](https://github.com/mohandshamada/AI_Drawings_Reader)
[![Coverage](https://img.shields.io/badge/coverage-20%25-yellow.svg)](https://github.com/mohandshamada/AI_Drawings_Reader)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AI Drawing Analyzer uses cutting-edge Vision-Language Models to extract text from complex PDFs with unprecedented accuracy. Perfect for technical drawings, blueprints, handwritten documents, and mixed-media layouts where traditional OCR falls short.

---

## ✨ Features

### Core Capabilities
- **🤖 AI-Powered OCR** - Vision-Language Models beat traditional OCR on complex documents
- **⚡ Resume Processing** - Interrupt and continue from last processed page
- **🔄 6 AI Providers** - Local inference + cloud APIs (OpenAI, Gemini, Claude, etc.)
- **💾 Multiple Output Formats** - JSONL, formatted text, and Toon format
- **📊 Real-time Progress** - Page-by-page status with tqdm progress bars
- **🛡️ Robust Error Handling** - Graceful recovery and detailed error messages

### Advanced Features (NEW)
- **📦 Toon Format Export** - Convert to space-efficient Toon format via Node.js bridge
- **⚡ Response Caching** - Cache API responses to reduce costs and improve speed
- **🧪 Test Suite** - 13 comprehensive unit tests with pytest
- **🔄 CI/CD Pipeline** - Automated testing on Python 3.9-3.13
- **🎨 Code Quality** - Linting (ruff), formatting (black), type checking (mypy)
- **📝 Comprehensive Docs** - Docstrings, contributing guide, and changelog

---

## 🚀 Quick Start

### Installation

#### Basic Installation
```bash
# Clone the repository
git clone https://github.com/mohandshamada/AI_Drawings_Reader.git
cd AI_Drawings_Reader

# Install the package
pip install -e .
```

#### With Local Models (Florence-2, Qwen, LLaVA)
```bash
pip install -e ".[local]"

# For GPU acceleration (10-100x faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### With Toon Format Support
```bash
# Install Node.js dependencies
pnpm install
# or
npm install

# Then install Python package
pip install -e .
```

#### Development Installation
```bash
# Install with all dev tools (testing, linting, formatting)
pip install -e ".[dev]"
```

### Your First Run

**Interactive Mode (Recommended)**
```bash
ai-drawing-analyzer your_document.pdf
```

The tool will:
1. Ask you to select an AI provider
2. Ask you to select a model
3. Start processing with real-time progress

**Direct Command**
```bash
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp
```

---

## 📖 Complete Usage Guide

### Basic Usage

#### Process a PDF
```bash
# Interactive mode
ai-drawing-analyzer document.pdf

# Specify provider and model
ai-drawing-analyzer document.pdf -p openai -m gpt-4o

# From URL
ai-drawing-analyzer https://example.com/drawing.pdf -p gemini -m gemini-2.0-flash-exp
```

#### Output Formats

**JSONL Output (default)**
```bash
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp -o results.jsonl
```

Output structure:
```json
{"page": 1, "page_type": "image", "provider": "gemini", "model": "gemini-2.0-flash-exp", "text_content": "...", "timestamp": "2025-01-30T..."}
{"page": 2, "page_type": "image", "provider": "gemini", "model": "gemini-2.0-flash-exp", "text_content": "...", "timestamp": "2025-01-30T..."}
```

**Convert to Text**
```bash
# Convert existing JSONL to readable text
ai-drawing-analyzer output.jsonl --to-text --output-text document.txt

# Or process and convert in one command
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp --to-text
```

Text output includes:
- Document metadata
- Cross-reference index
- Legend & symbols
- Complete page-by-page content

**Convert to Toon Format** ✨ NEW
```bash
# Requires Node.js and pnpm/npm install

# Process and convert to Toon format
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp --to-toon

# Custom Toon output path
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp --to-toon --output-toon results.toon

# Convert existing JSONL to Toon
node scripts/convert_to_toon.mjs output.jsonl output.toon
```

### Advanced Features

#### Resume Processing
```bash
# Start processing
ai-drawing-analyzer large_doc.pdf -p gemini -m gemini-2.0-flash-exp -o output.jsonl

# If interrupted, resume from last page
ai-drawing-analyzer large_doc.pdf -p gemini -m gemini-2.0-flash-exp -o output.jsonl --resume
```

#### High-Precision Mode
```bash
# Increase zoom for better text detection (1-4)
export PDF_ZOOM_LEVEL=3
export JPEG_QUALITY=95

ai-drawing-analyzer technical_drawing.pdf -p huggingface-local -m microsoft/Florence-2-large
```

#### Batch Processing
```bash
#!/bin/bash
# Process all PDFs in a directory

for pdf in *.pdf; do
  echo "Processing $pdf..."
  ai-drawing-analyzer "$pdf" \
    -p gemini -m gemini-2.0-flash-exp \
    -o "${pdf%.pdf}_output.jsonl" \
    --resume --to-text --to-toon
done
echo "✅ All documents processed!"
```

---

## 🎛️ CLI Options Reference

### Positional Arguments
```
pdf                     PDF file path, URL, or JSONL file for conversion
```

### Optional Arguments

#### Provider & Model Selection
```
-p, --provider PROVIDER
                        AI provider: huggingface-local, huggingface, openrouter,
                        gemini, openai, anthropic (default: interactive)

-m, --model MODEL       Model ID (default: interactive selection)

-k, --api-key API_KEY   API key (default: from environment)
```

#### Input/Output Options
```
-e, --env ENV           Path to .env file (default: .env)

-o, --output OUTPUT     Output JSONL file path (default: auto-generated timestamp)

--resume                Resume processing from last completed page
```

#### Format Conversion
```
--to-text               Convert JSONL to formatted text

--output-text PATH      Output text file path (default: drawing_complete.txt)

--to-toon               Convert output to Toon format (requires Node.js)

--output-toon PATH      Output Toon file path (default: auto-generated)
```

#### Help
```
-h, --help              Show help message and exit
```

---

## 🤖 Available Providers & Models

### Local Inference (Free, No API Key)

**Provider:** `huggingface-local`

| Model | Quality | Speed | VRAM | Best For |
|-------|---------|-------|------|----------|
| microsoft/Florence-2-large | ⭐⭐⭐⭐⭐ | ⭐⭐ | 8GB | Technical drawings, blueprints |
| microsoft/Florence-2-base | ⭐⭐⭐⭐ | ⭐⭐⭐ | 4GB | General documents |
| Qwen/Qwen2-VL-7B-Instruct | ⭐⭐⭐⭐ | ⭐⭐ | 16GB | Mixed-media documents |
| llava-hf/llava-1.5-7b-hf | ⭐⭐⭐ | ⭐⭐⭐ | 8GB | General purpose |

**Setup:**
```bash
pip install -e ".[local]"

# Usage
ai-drawing-analyzer doc.pdf -p huggingface-local -m microsoft/Florence-2-large
```

### Cloud APIs

#### Google Gemini (Free Tier Available)

**Provider:** `gemini`
**API Key:** `GOOGLE_API_KEY`
**Free Tier:** 60 requests/minute

| Model | Context | Speed | Cost |
|-------|---------|-------|------|
| gemini-2.0-flash-exp | 1M tokens | ⭐⭐⭐⭐⭐ | Free (experimental) |
| gemini-1.5-pro | 2M tokens | ⭐⭐⭐⭐ | Paid |
| gemini-1.5-flash | 1M tokens | ⭐⭐⭐⭐⭐ | Low cost |

**Setup:**
```bash
export GOOGLE_API_KEY="your_key_here"
ai-drawing-analyzer doc.pdf -p gemini -m gemini-2.0-flash-exp
```

#### OpenAI

**Provider:** `openai`
**API Key:** `OPENAI_API_KEY`

| Model | Quality | Speed | Cost |
|-------|---------|-------|------|
| gpt-4o | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$ |
| gpt-4o-mini | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $ |

**Setup:**
```bash
export OPENAI_API_KEY="sk-..."
ai-drawing-analyzer doc.pdf -p openai -m gpt-4o
```

#### Anthropic Claude

**Provider:** `anthropic`
**API Key:** `ANTHROPIC_API_KEY`

| Model | Quality | Speed | Cost |
|-------|---------|-------|------|
| claude-3-5-sonnet-20241022 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$ |
| claude-3-5-haiku-20241022 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $ |

**Setup:**
```bash
export ANTHROPIC_API_KEY="your_key_here"
ai-drawing-analyzer doc.pdf -p anthropic -m claude-3-5-sonnet-20241022
```

#### HuggingFace Router (Free Tier)

**Provider:** `huggingface`
**API Key:** `HF_TOKEN`

| Model | Notes |
|-------|-------|
| Qwen/Qwen2.5-VL-7B-Instruct | Recommended, well-supported |
| mistralai/Pixtral-12B-2409 | Backup option |
| nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 | Backup option |

**Setup:**
```bash
export HF_TOKEN="hf_..."
ai-drawing-analyzer doc.pdf -p huggingface -m Qwen/Qwen2.5-VL-7B-Instruct
```

#### OpenRouter (Multi-Model Gateway)

**Provider:** `openrouter`
**API Key:** `OPENROUTER_API_KEY`

Access to: Claude, GPT-4o, Gemini, Llama 3.2 Vision, and more

**Setup:**
```bash
export OPENROUTER_API_KEY="your_key_here"
ai-drawing-analyzer doc.pdf -p openrouter -m anthropic/claude-3.5-sonnet
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project directory:

```bash
# API Keys
GOOGLE_API_KEY=your_gemini_key_here
OPENAI_API_KEY=sk-your_key_here
ANTHROPIC_API_KEY=your_key_here
HF_TOKEN=hf_your_token_here
OPENROUTER_API_KEY=your_key_here

# PDF Processing
PDF_ZOOM_LEVEL=2              # 1-4 (higher = better detail, slower)
JPEG_QUALITY=90               # 1-100 (higher = better quality)

# Network
API_TIMEOUT=120               # Seconds for API calls
DOWNLOAD_TIMEOUT=60           # Seconds for PDF downloads
MAX_RETRIES=3                 # Number of retry attempts

# LLM
MAX_TOKENS=2048               # Max response length

# Debug
DEBUG=false                   # Set to true for detailed error traces
```

### Configuration Priority

1. Command-line arguments (highest priority)
2. Environment variables
3. `.env` file
4. Default values (lowest priority)

**Example:**
```bash
# Override with command-line
ai-drawing-analyzer doc.pdf -p gemini -m gemini-2.0-flash-exp --api-key "key_override"

# Or use .env file
ai-drawing-analyzer doc.pdf -p gemini -m gemini-2.0-flash-exp --env custom.env
```

---

## 📚 Tutorials & Examples

### Tutorial 1: Extract Text from Blueprint

```bash
# 1. Use high precision settings for technical drawings
export PDF_ZOOM_LEVEL=3
export JPEG_QUALITY=95

# 2. Process with Florence-2 (best for technical drawings)
ai-drawing-analyzer blueprint.pdf \
  -p huggingface-local \
  -m microsoft/Florence-2-large \
  -o blueprint_ocr.jsonl

# 3. Convert to readable text
ai-drawing-analyzer blueprint_ocr.jsonl \
  --to-text \
  --output-text blueprint_complete.txt

# 4. Output includes:
#    - Metadata (title, scale, revision)
#    - Cross-references
#    - Legends and symbols
#    - Page-by-page content
```

### Tutorial 2: Batch Process Documents

```bash
# Create processing script
cat > batch_process.sh << 'EOF'
#!/bin/bash

PROVIDER="gemini"
MODEL="gemini-2.0-flash-exp"

for pdf in documents/*.pdf; do
  filename=$(basename "$pdf" .pdf)
  echo "📄 Processing: $filename"

  ai-drawing-analyzer "$pdf" \
    -p "$PROVIDER" \
    -m "$MODEL" \
    -o "output/${filename}.jsonl" \
    --resume \
    --to-text \
    --output-text "output/${filename}.txt"

  echo "✅ Completed: $filename"
done

echo "🎉 All documents processed!"
EOF

chmod +x batch_process.sh
./batch_process.sh
```

### Tutorial 3: Resume Interrupted Processing

```bash
# Start processing large document
ai-drawing-analyzer manual_500pages.pdf \
  -p gemini \
  -m gemini-2.0-flash-exp \
  -o manual.jsonl

# If interrupted (Ctrl+C, network error, etc.)
# Pages 1-247 are saved in manual.jsonl

# Resume from page 248
ai-drawing-analyzer manual_500pages.pdf \
  -p gemini \
  -m gemini-2.0-flash-exp \
  -o manual.jsonl \
  --resume

# The tool automatically detects completed pages
# and continues from where it left off
```

### Tutorial 4: Compare AI Providers

```bash
# Test same document with different providers
DOCUMENT="test_drawing.pdf"

# Local inference (free, slower)
ai-drawing-analyzer "$DOCUMENT" \
  -p huggingface-local \
  -m microsoft/Florence-2-large \
  -o test_florence.jsonl

# Google Gemini (fast, free tier)
ai-drawing-analyzer "$DOCUMENT" \
  -p gemini \
  -m gemini-2.0-flash-exp \
  -o test_gemini.jsonl

# OpenAI GPT-4o (highest quality, paid)
ai-drawing-analyzer "$DOCUMENT" \
  -p openai \
  -m gpt-4o \
  -o test_gpt4o.jsonl

# Compare outputs
diff test_florence.jsonl test_gemini.jsonl
```

### Tutorial 5: Export to Toon Format

```bash
# Requires Node.js
pnpm install

# Process and export to Toon format
ai-drawing-analyzer document.pdf \
  -p gemini \
  -m gemini-2.0-flash-exp \
  --to-toon

# This creates:
# - output_ocr_20250130_123456.jsonl (OCR results)
# - output_ocr_20250130_123456.toon (Toon format)

# Toon format is more compact and structured
# Perfect for further processing with Toon-compatible tools
```

---

## 🛠️ Development

### Development Setup

```bash
# Clone repository
git clone https://github.com/mohandshamada/AI_Drawings_Reader.git
cd AI_Drawings_Reader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install Node.js dependencies (for Toon format)
pnpm install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ai_drawing_analyzer --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

### Code Quality Checks

```bash
# Lint with ruff
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format with black
black src/ tests/

# Check formatting (without changes)
black --check src/ tests/

# Type check with mypy
mypy src/

# Run all checks at once
pytest && ruff check src/ && black --check src/ && mypy src/
```

### Project Structure

```
AI_Drawings_Reader/
├── .github/
│   └── workflows/
│       └── test.yml           # CI/CD pipeline
├── src/
│   └── ai_drawing_analyzer/
│       ├── clients/           # AI provider implementations
│       │   ├── base.py        # Abstract base client
│       │   ├── factory.py     # Client factory
│       │   ├── gemini.py      # Google Gemini
│       │   ├── openai.py      # OpenAI & OpenRouter
│       │   ├── anthropic.py   # Anthropic Claude
│       │   └── huggingface.py # HF Router & Local
│       ├── converters/        # Format converters
│       │   ├── jsonl_to_text.py
│       │   └── toon_converter.py
│       ├── processing/        # PDF processing
│       │   └── pdf.py
│       ├── utils/             # Utilities
│       │   ├── cache.py       # Response caching
│       │   ├── config.py      # Configuration
│       │   ├── files.py       # File handling
│       │   └── logging.py     # Logging setup
│       └── cli.py             # CLI interface
├── tests/                     # Test suite
│   ├── conftest.py            # Shared fixtures
│   ├── test_config.py
│   ├── test_pdf_processing.py
│   └── test_clients/
│       └── test_factory.py
├── scripts/
│   └── convert_to_toon.mjs    # Toon converter script
├── legacy/                    # Deprecated code
│   ├── README.md
│   └── dwg_analyzer.py
├── pyproject.toml             # Project config
├── package.json               # Node.js deps
├── CHANGELOG.md               # Version history
├── CONTRIBUTING.md            # Dev guidelines
└── README.md                  # This file
```

---

## 🐛 Troubleshooting

### Common Issues

#### "Command not found: ai-drawing-analyzer"

**Solution:**
```bash
# Reinstall package
pip install -e .

# Check installation
which ai-drawing-analyzer
ai-drawing-analyzer --help
```

#### "Missing API Key" Error

**Solution:**
```bash
# Option 1: Environment variable
export GOOGLE_API_KEY="your_key"
ai-drawing-analyzer doc.pdf -p gemini -m gemini-2.0-flash-exp

# Option 2: Command-line flag
ai-drawing-analyzer doc.pdf -p gemini -m gemini-2.0-flash-exp -k "your_key"

# Option 3: .env file
echo "GOOGLE_API_KEY=your_key" >> .env
ai-drawing-analyzer doc.pdf -p gemini -m gemini-2.0-flash-exp
```

#### "Transformers and torch required for local inference"

**Solution:**
```bash
# Install local dependencies
pip install -e ".[local]"

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Processing is Too Slow

**Solution:**
```bash
# Use lower quality settings
export PDF_ZOOM_LEVEL=1
export JPEG_QUALITY=70

# Or use faster cloud provider
ai-drawing-analyzer doc.pdf -p gemini -m gemini-2.0-flash-exp
```

#### Network Timeout Errors

**Solution:**
```bash
# Increase timeout
export API_TIMEOUT=300
export DOWNLOAD_TIMEOUT=180

ai-drawing-analyzer doc.pdf -p openai -m gpt-4o
```

#### Toon Conversion Fails

**Solution:**
```bash
# Check Node.js installation
node --version

# Install dependencies
pnpm install
# or
npm install

# Verify script exists
ls scripts/convert_to_toon.mjs

# Try manual conversion
node scripts/convert_to_toon.mjs input.jsonl output.toon
```

#### Low Quality OCR Results

**Solution:**
```bash
# Increase resolution
export PDF_ZOOM_LEVEL=3
export JPEG_QUALITY=95

# Use best model for document type
# For technical drawings:
ai-drawing-analyzer doc.pdf -p huggingface-local -m microsoft/Florence-2-large

# For general documents:
ai-drawing-analyzer doc.pdf -p openai -m gpt-4o
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Contribution Guide

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Make your changes**
   - Add tests for new features
   - Follow code style (black, ruff)
   - Update documentation
4. **Run tests and checks**
   ```bash
   pytest
   ruff check src/ tests/
   black --check src/ tests/
   ```
5. **Commit and push**
   ```bash
   git commit -m "feat: add new feature"
   git push origin feature/your-feature
   ```
6. **Create Pull Request**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Vision Language Models:** Florence-2, Qwen-VL, GPT-4o, Claude 3.5, Gemini
- **Libraries:** PyMuPDF, Pillow, httpx, transformers, torch
- **Testing:** pytest, pytest-cov, pytest-asyncio
- **Code Quality:** ruff, black, mypy
- **Toon Format:** @toon-format/toon

---

## 📞 Support & Community

- **Documentation:** See [QUICK_START.md](QUICK_START.md) for rapid setup
- **Enhancements:** Check [ENHANCEMENTS_SUMMARY.md](ENHANCEMENTS_SUMMARY.md) for latest features
- **Issues:** [GitHub Issues](https://github.com/mohandshamada/AI_Drawings_Reader/issues)
- **Discussions:** [GitHub Discussions](https://github.com/mohandshamada/AI_Drawings_Reader/discussions)

---

## 🎯 Roadmap

- [ ] Support for more output formats (Markdown, HTML)
- [ ] Parallel page processing
- [ ] Web interface
- [ ] Docker container
- [ ] Batch API for high-volume processing
- [ ] Custom model fine-tuning guides

---

**Made with ❤️ by the AI Drawing Analyzer team**

**Version:** 2.2.0+ (claude-Manjaro branch)
**Last Updated:** January 2025
**Status:** ✅ Actively Maintained

---

## Star History

If you find this project useful, please consider giving it a star ⭐ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=mohandshamada/AI_Drawings_Reader&type=Date)](https://star-history.com/#mohandshamada/AI_Drawings_Reader&Date)
