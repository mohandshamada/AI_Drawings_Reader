# AI Drawing Analyzer

**Extract text from PDFs using AI Vision Models — No traditional OCR needed.**

The AI Drawing Analyzer uses multimodal vision-language models to extract text from complex PDFs with unmatched accuracy. Perfect for technical drawings, blueprints, handwritten documents, and mixed-media layouts where traditional OCR fails.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version 2.2.0](https://img.shields.io/badge/Version-2.2.0-brightgreen.svg)](#)

---

## ✨ Key Features

- **🤖 AI-Powered OCR** — Uses Vision-Language Models for superior text extraction
- **⚡ Resume Processing** — Continue from last completed page after interruptions
- **🔄 6 AI Providers** — Local inference or cloud APIs (free tiers available)
- **⚙️ Fully Configurable** — Control zoom, quality, timeouts, and more
- **📊 Real-time Progress** — Page-by-page status tracking
- **🛡️ Robust Error Handling** — Input/response validation with graceful recovery
- **📁 Batch Processing** — Process multiple PDFs efficiently
- **💾 Multiple Output Formats** — JSON, JSONL, or formatted text

---

## 🚀 Quick Start

### Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/mohandshamada/AI_Drawings_Reader.git
cd AI_Drawings_Reader

# Install the package
pip install -e .

# For local models (Florence-2, Qwen, etc.)
pip install -e ".[local]"

# For GPU acceleration (10-100x faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Your First Run

```bash
# Interactive mode (recommended)
ai-drawing-analyzer your_document.pdf
```

The tool will:
1. Ask you to select a provider
2. Ask you to select a model
3. Start processing with real-time progress

**That's it!** No configuration needed.

---

## 📋 Usage Examples

### Option 1: Interactive Mode ✨
*Best for first time or when you're unsure*

```bash
ai-drawing-analyzer document.pdf
```

### Option 2: Direct Command 🎯
*Best for automation or scripts*

```bash
ai-drawing-analyzer document.pdf -p huggingface-local -m microsoft/Florence-2-large
```

### Common Examples

**Local Processing (No API key):**
```bash
# Florence-2 Large — Best OCR quality
ai-drawing-analyzer drawing.pdf -p huggingface-local -m microsoft/Florence-2-large

# Qwen2-VL 7B — Faster, good quality
ai-drawing-analyzer drawing.pdf -p huggingface-local -m Qwen/Qwen2-VL-7B-Instruct
```

**Free Cloud APIs:**
```bash
# Google Gemini (60 req/min free)
ai-drawing-analyzer drawing.pdf -p gemini -m gemini-2.0-flash-exp

# HuggingFace Router (free)
ai-drawing-analyzer drawing.pdf -p huggingface -m Qwen/Qwen2.5-VL-7B-Instruct
```

**Paid Cloud APIs:**
```bash
# OpenAI GPT-4o
ai-drawing-analyzer drawing.pdf -p openai -m gpt-4o

# Anthropic Claude 3.5 Sonnet
ai-drawing-analyzer drawing.pdf -p anthropic -m claude-3-5-sonnet-20241022
```

**Advanced Features:**
```bash
# Resume after interruption
ai-drawing-analyzer drawing.pdf -p gemini -m gemini-2.0-flash-exp --resume

# Custom output file
ai-drawing-analyzer drawing.pdf -p gemini -m gemini-2.0-flash-exp -o my_output.jsonl

# Process URL + convert to text
ai-drawing-analyzer https://example.com/drawing.pdf \
  -p huggingface-local \
  -m microsoft/Florence-2-large \
  --to-text

# High-precision mode (3x zoom)
export PDF_ZOOM_LEVEL=3
ai-drawing-analyzer technical_drawing.pdf -p huggingface-local -m microsoft/Florence-2-large
```

---

## 🔧 Configuration

### Environment Variables

Control processing behavior without command-line arguments:

```bash
# PDF Processing
export PDF_ZOOM_LEVEL=2              # 1-4 (higher = better detail, slower)
export JPEG_QUALITY=90               # 1-100 (higher = better quality, slower upload)

# Network
export API_TIMEOUT=120               # Seconds (for API calls)
export DOWNLOAD_TIMEOUT=60           # Seconds (for PDF download)
export MAX_RETRIES=3                 # Number of retry attempts

# LLM
export MAX_TOKENS=2048               # Max response length
```

### .env File

Create `.env` in your project directory:

```env
# AI Provider API Keys
GOOGLE_API_KEY=your_gemini_key_here
OPENAI_API_KEY=sk-your_key_here
ANTHROPIC_API_KEY=your_key_here
HF_TOKEN=hf_your_token_here

# Optional: Processing Settings
PDF_ZOOM_LEVEL=2
JPEG_QUALITY=90
API_TIMEOUT=120
```

Settings auto-load when you run the tool!

---

## 📊 Available Providers & Models

| Provider | Setup | Models | Speed | Cost |
|----------|-------|--------|-------|------|
| **HuggingFace Local** | `pip install -e ".[local]"` | Florence-2, Qwen-VL, LLaVA | ⭐⭐⭐ (GPU) | Free |
| **Google Gemini** | `export GOOGLE_API_KEY=...` | Gemini 2.0 Flash, 1.5 Pro | ⭐⭐ | Free tier |
| **OpenAI** | `export OPENAI_API_KEY=...` | GPT-4o, GPT-4o Mini | ⭐⭐ | Paid |
| **Anthropic** | `export ANTHROPIC_API_KEY=...` | Claude 3.5 Sonnet/Haiku | ⭐⭐ | Paid |
| **HuggingFace Router** | `export HF_TOKEN=...` | Qwen, Pixtral, Nemotron | ⭐⭐ | Free tier |
| **OpenRouter** | `export OPENROUTER_API_KEY=...` | Claude, GPT-4o, Gemini | ⭐⭐ | Paid |

---

## 🎯 Real-World Examples

### Example 1: Resume Processing
```bash
# Start processing (interrupted after page 42)
ai-drawing-analyzer large_document.pdf -p gemini -m gemini-2.0-flash-exp

# Later, resume from page 43
ai-drawing-analyzer large_document.pdf -p gemini -m gemini-2.0-flash-exp --resume
```

### Example 2: Batch Process Multiple PDFs
```bash
#!/bin/bash
for pdf in *.pdf; do
  echo "Processing $pdf..."
  ai-drawing-analyzer "$pdf" \
    -p huggingface-local \
    -m microsoft/Florence-2-large \
    -o "${pdf%.pdf}_output.jsonl" \
    --resume
done
echo "✅ All documents processed!"
```

### Example 3: High-Precision Technical Drawings
```bash
# 3x zoom + high quality for small text
export PDF_ZOOM_LEVEL=3
export JPEG_QUALITY=95
ai-drawing-analyzer technical_drawing.pdf \
  -p huggingface-local \
  -m microsoft/Florence-2-large
```

### Example 4: Fast Processing
```bash
# 1x zoom + low quality for quick processing
export PDF_ZOOM_LEVEL=1
export JPEG_QUALITY=70
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp
```

### Example 5: Convert OCR to LLM-Ready Text
```bash
# Extract from PDF and convert to readable text
ai-drawing-analyzer blueprint.pdf \
  -p huggingface-local \
  -m microsoft/Florence-2-large \
  --to-text \
  --output-text blueprint_complete.txt
```

---

## 🆘 Troubleshooting

### "Command not found: ai-drawing-analyzer"
```bash
# Reinstall the package
pip install -e .
```

### "Missing API Key" Error
```bash
# Option 1: Set environment variable
export OPENAI_API_KEY=sk-...
ai-drawing-analyzer document.pdf -p openai -m gpt-4o

# Option 2: Use command-line flag
ai-drawing-analyzer document.pdf -p openai -m gpt-4o --api-key sk-...

# Option 3: Create .env file
echo "OPENAI_API_KEY=sk-..." > .env
ai-drawing-analyzer document.pdf -p openai -m gpt-4o
```

### "Transformers and torch required for local inference"
```bash
# Install local model dependencies
pip install -e ".[local]"

# For GPU (much faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Processing is Slow
```bash
# Use lower zoom level and quality
export PDF_ZOOM_LEVEL=1
export JPEG_QUALITY=70
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp
```

### Network Timeouts
```bash
# Increase timeout
export API_TIMEOUT=180
export DOWNLOAD_TIMEOUT=120
ai-drawing-analyzer large_document.pdf -p openai -m gpt-4o
```

---

## 📚 What's New (v2.2.0)

✨ **Resume Capability** — Continue processing after interruptions
⚙️ **Configuration Management** — Full environment variable support
🛡️ **Enhanced Error Handling** — Input/response validation on all APIs
📝 **New CLI Options** — `--output`, `--resume`, `--env` flags
💪 **Advanced Features** — Fine-tuned rendering, batch processing examples

See [ENHANCEMENTS.md](ENHANCEMENTS.md) for detailed changelog.

---

## 📖 Command Reference

```bash
# Show all options
ai-drawing-analyzer --help

# Interactive mode (recommended)
ai-drawing-analyzer document.pdf

# Specify provider and model
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp

# Resume from last page
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp --resume

# Custom output file
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp -o my_output.jsonl

# Convert existing JSONL to text
ai-drawing-analyzer output.jsonl --to-text -o document_complete.txt

# Specify custom .env file
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp --env /path/to/.env
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

- 📖 **Documentation**: Check [QUICK_START.md](QUICK_START.md) for rapid setup
- 🔍 **Issues**: See [ENHANCEMENTS.md](ENHANCEMENTS.md) for known improvements
- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Suggestions**: Discussions welcome!

---

**Last Updated**: January 2025
**Version**: 2.2.0
**Status**: Actively Maintained ✅
