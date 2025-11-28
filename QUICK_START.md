# Quick Start Guide - AI Drawing Analyzer Complete Edition

## 🚀 Fastest Path: Automated Installation Scripts ⭐

### The Easiest Way to Get Started

We provide installation scripts that do everything for you with an interactive setup:

**On macOS/Linux:**
```bash
chmod +x install.sh
./install.sh
```

**On Windows:**
```bash
# Simply double-click or run:
install.bat
```

The scripts will:
- Check Python installation
- Create virtual environment
- Ask you to choose: Local Models, Cloud APIs, or Both
- Install all dependencies automatically
- Provide next steps and GPU acceleration tips

This is the recommended approach - let the script handle setup automatically!

---

## 🚀 Manual Setup (Choose Your Path)

### Path 1: Local Models (NO API Key Needed) ⭐ Recommended

```bash
# Install (one-time setup)
pip install -r requirements-local.txt

# Run with Florence-2-large locally
ai-drawing-analyzer your_drawing.pdf -p huggingface-local -m microsoft/Florence-2-large
```

**Why local?** Free, no API keys needed, works offline, excellent OCR quality ✨

### Path 2: Cloud APIs (API Key Required)

```bash
# Install (minimal dependencies)
pip install -r requirements-minimal.txt

# Run with Google Gemini (free tier)
ai-drawing-analyzer your_drawing.pdf -p gemini
```

**Why cloud?** Faster for first use, no GPU needed, access to multiple models

---

## 📋 Manual Installation Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/mohandshamada/AI-Drawings-Reader.git
cd AI-Drawings-Reader
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate:
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate          # Windows
```

### Step 3: Install Dependencies

**For Local Models (Florence-2, Qwen-VL, etc.):**
```bash
pip install -r requirements-local.txt

# For GPU support (CUDA 11.8 - much faster):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For Cloud APIs Only:**
```bash
pip install -r requirements-minimal.txt
```

### Step 4: Configure API Keys (Cloud APIs Only)

**Skip this if using local models!**

Create `.env` file:
```bash
# Copy the example
cp .env.example .env

# Edit .env and uncomment ONE or more API keys:
# GOOGLE_API_KEY=AIzaxxxxxxxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxxxxxxx
# HF_TOKEN=hf_xxxxxxxxxxxxx
```

---

## 🎯 Usage Examples

### Interactive Mode (Recommended)
```bash
ai-drawing-analyzer document.pdf
# Select provider and model from menu
```

### Command Line (Fastest)

**Local Florence-2:**
```bash
ai-drawing-analyzer document.pdf -p huggingface-local -m microsoft/Florence-2-large
```

**Local Qwen-VL:**
```bash
ai-drawing-analyzer document.pdf -p huggingface-local -m Qwen/Qwen2-VL-7B-Instruct
```

**Local Qwen3-VL (Advanced):**
```bash
ai-drawing-analyzer document.pdf -p huggingface-local -m Qwen/Qwen3-VL-235B-A22B-Thinking
```

**Cloud - Google Gemini (free):**
```bash
ai-drawing-analyzer document.pdf -p gemini
```

**Cloud - OpenAI GPT-4o:**
```bash
ai-drawing-analyzer document.pdf -p openai -m gpt-4o
```

**Remote PDF URL:**
```bash
ai-drawing-analyzer https://example.com/drawing.pdf -p huggingface-local
```

---

## 💡 Tips & Tricks

### Tip 1: Test Installation
```bash
# Show help
ai-drawing-analyzer --help

# Test with a small PDF (1-2 pages)
ai-drawing-analyzer test.pdf -p huggingface-local -m microsoft/Florence-2-base
```

### Tip 2: First Run Takes Time
- **Local models:** First run downloads model (~2-24 GB, 5-10 minutes depending on model)
- **Cloud APIs:** Instant (after API setup)

### Tip 3: Output Format
Results saved as `output_ocr_YYYYMMDD_HHMMSS.json` (newline-delimited JSON):
```json
{"page": 1, "text_content": "OCR extracted text...", ...}
{"page": 2, "text_content": "More text...", ...}
```

### Tip 4: Memory Issues?
```bash
# Use smaller model
ai-drawing-analyzer document.pdf -p huggingface-local -m microsoft/Florence-2-base

# Or CPU-only mode
export CUDA_VISIBLE_DEVICES=""
ai-drawing-analyzer document.pdf -p huggingface-local
```

---

## 🆓 Free API Keys for Cloud Services

| Service | Free Tier | Link |
|---------|-----------|------|
| Google Gemini | 60 req/min | https://makersuite.google.com/app/apikey |
| HuggingFace | Free | https://huggingface.co/settings/tokens |
| OpenAI | $5 credit | https://platform.openai.com/api-keys |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: transformers` | `pip install transformers torch` |
| CUDA Out of Memory | Use smaller model or CPU mode (see Tip 4) |
| `Model Not Supported` | Use `huggingface-local` instead of `huggingface` |
| API key not found | Check `.env` format: `KEY=value` (no spaces/quotes) |
| Download fails | Ensure URL is publicly accessible |

---

## 📖 Documentation

- **Full Guide:** See `README.md` for complete documentation
- **Supported Models:** All Hugging Face vision models
- **Advanced Config:** Environment variables, custom cache directory, etc.

---

## ✨ What Happens on First Run?

1. **Local models:** Downloads model from Hugging Face (one-time)
2. **Cloud APIs:** Fetches available models from provider
3. **Processing:** Converts PDF → high-res images → extracts text via AI
4. **Output:** Saves results as JSON (one entry per page)

---

## 🎉 That's It!

You're ready to analyze PDFs with AI:
- No background in ML required
- Works with any vision model from Hugging Face
- Free options available (local or cloud)

**Quick command to test:**
```bash
ai-drawing-analyzer your_file.pdf -p huggingface-local -m microsoft/Florence-2-base
```

For detailed help, see `README.md` or run:
```bash
ai-drawing-analyzer --help
```

Happy analyzing! 🚀
