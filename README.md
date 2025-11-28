# AI Drawing Analyzer – Complete Edition

A command-line tool that leverages Vision-Language Models (VLMs) to perform high-quality OCR and analysis on PDF documents.
Unlike traditional OCR, this tool converts PDF pages into high-resolution images and uses advanced AI models to "read" the text — making it significantly more accurate for complex layouts, handwriting, technical drawings, and blueprints.

---

## 🚀 Key Features

- **Smart Vision OCR** — Uses multimodal AI for OCR, surpassing standard OCR on complex documents (blueprints, handwritten notes, mixed text & images).
- **Local & Cloud Providers** — Run models locally or via cloud APIs:
  - **HuggingFace Local** (NEW!) — Run **any** HF vision model locally (Florence-2, Qwen-VL, LLaVA, BLIP-2, etc.)
  - **HuggingFace Router** — Cloud-based (Qwen2.5-VL, Pixtral, Nemotron, etc.)
  - **Google Gemini** — Includes built-in retry logic for rate-limit / quota errors
  - **OpenRouter / OpenAI / Anthropic** — Access to GPT-4o, Claude 3.5 Sonnet, etc.
- **Universal Input** — Accepts local file paths or public URLs
- **High-Resolution Processing** — PDF → image conversion at 2× resolution for maximum detail
- **Robust Error Handling** — Gracefully handles download failures, API errors, rate limits
- **Structured Output** — Saves results as newline-delimited JSON (`.jsonl`)
- **Modular Architecture** — Refactored codebase for better maintainability and extensibility
- **Async/Await Support** — Faster processing for API calls with parallel capabilities
- **Progress Bars** — Real-time progress tracking for large documents using `tqdm`

---

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- Internet connection (for downloading PDFs or using cloud APIs)
- Git (optional, for cloning the repository)

### Quick Start with Installation Script (Recommended) ⭐

We provide fully automated installation scripts for both Windows and Linux/macOS that handle everything:

#### On Windows:
```bash
# Simply double-click or run from command prompt:
install.bat
```

#### On macOS/Linux:
```bash
# Make the script executable and run it:
chmod +x install.sh
./install.sh
```

#### What the Scripts Do:

The installation scripts provide an **interactive setup experience** with three clear options:

**Option 1️⃣  LOCAL MODELS ONLY** (Recommended)
- Run Florence-2, Qwen-VL, LLaVA locally on your machine
- No API keys required
- Best for: Technical drawings, blueprints, offline usage
- Install: `requirements-local.txt`
- GPU acceleration available for 10-100x faster inference

**Option 2️⃣  CLOUD APIs ONLY** (Minimal Setup)
- Use Google Gemini, OpenAI, Claude, etc.
- Requires API keys (free tiers available)
- Best for: Quick testing, occasional use, no GPU needed
- Install: `requirements-minimal.txt`

**Option 3️⃣  BOTH LOCAL & CLOUD** (Full Setup)
- All features: local models + cloud APIs
- Choose which provider to use at runtime
- Best for: Flexibility and experimentation

#### Script Features:
1. ✅ Check Python installation
2. ✅ Create a virtual environment
3. ✅ Interactive choice: Local, Cloud, or Both (defaults to Local)
4. ✅ Install appropriate dependencies
5. ✅ Create `.env` file from template
6. ✅ Provide GPU acceleration setup instructions (CUDA 11.8)
7. ✅ Show next steps specific to your choice
8. ✅ Display helpful tips and documentation references

---

### Manual Installation Steps

#### Step 1: Clone/Download the Repository

```bash
git clone https://github.com/mohandshamada/AI-Drawings-Reader.git
cd AI-Drawings-Reader
```

#### Step 2: Install uv (if not already installed)

```bash
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or download from [astral.sh/uv](https://astral.sh/uv)

#### Step 3: Create a Virtual Environment and Install Dependencies

Choose the installation option that matches your use case:

**Option A: Local Models Only (Florence-2, Qwen-VL, LLaVA)** (Recommended)
```bash
# Create virtual environment and install dependencies
uv sync --extra local
```

**Option B: Cloud APIs Only (No local GPU needed)**
```bash
# Create virtual environment with minimal dependencies
uv sync
```

**Option C: Both Local & Cloud (Full Installation)**
```bash
# Create virtual environment with all dependencies
uv sync --all-extras
```

**Option D: Local Models with GPU Acceleration (CUDA 11.8)**
```bash
# Install base dependencies with uv
uv sync --extra local

# Then install PyTorch with CUDA support (much faster!)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For other CUDA versions, see [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

#### Step 4: Configure API Keys (Only needed for Cloud Providers)

**Note:** Skip this if using only local models!

Copy and edit the `.env` file:

```bash
# Copy template
cp .env.example .env

# Edit with your favorite editor (nano, vim, code, etc.)
nano .env
```

Add your API keys:

```env
# HuggingFace Router (Free tier available)
HF_TOKEN=hf_your_token_here

# Google Gemini (Optional - Free tier: 60 req/min)
GOOGLE_API_KEY=your_gemini_key_here

# OpenRouter (Optional - Paid)
OPENROUTER_API_KEY=your_key_here

# OpenAI (Optional - Paid after free credits)
OPENAI_API_KEY=your_key_here

# Anthropic (Optional - Paid)
ANTHROPIC_API_KEY=your_key_here
```

**Note:** If using only local models (HuggingFace Local), no API keys are needed!

---

## 📖 Usage

### Interactive Mode (Recommended for beginners)

```bash
ai-drawing-analyzer /path/to/your/document.pdf
```

The script will guide you through:
1. Selecting a provider (Local, Cloud API, etc.)
2. Choosing a model
3. Processing the PDF

### Command-Line Mode

```bash
# With Florence-2-large locally
ai-drawing-analyzer document.pdf -p huggingface-local -m microsoft/Florence-2-large

# With Qwen-VL 7B locally
ai-drawing-analyzer document.pdf -p huggingface-local -m Qwen/Qwen2-VL-7B-Instruct

# With HuggingFace Router API (free)
ai-drawing-analyzer document.pdf -p huggingface -m Qwen/Qwen2.5-VL-7B-Instruct

# With Google Gemini
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp

# With OpenAI GPT-4o
ai-drawing-analyzer document.pdf -p openai -m gpt-4o

# With Claude 3.5 Sonnet
ai-drawing-analyzer document.pdf -p anthropic -m claude-3-5-sonnet-20241022
```

### Using a Remote PDF URL

```bash
ai-drawing-analyzer https://example.com/document.pdf -p huggingface-local -m microsoft/Florence-2-large
```

---

## 🎓 Detailed Tutorial

### Tutorial 1: First Run with Florence-2 (Local, No API Key)

**Estimated time:** 10-15 minutes (depending on internet speed)

#### Step 1: Prepare

```bash
# Activate virtual environment
source venv/bin/activate    # Linux/macOS
# or
venv\Scripts\activate       # Windows

# Verify installation
ai-drawing-analyzer --help
```

#### Step 2: Run Interactively

```bash
ai-drawing-analyzer electrical_drawing.pdf
```

#### Step 3: Follow the Prompts

```
🚀 AI Drawings Reader
======================================================================

🤖 Available AI Providers:
==================================================
1. Huggingface-Local
2. Huggingface
3. Openrouter
4. Gemini
5. Openai
6. Anthropic
==================================================

Select provider (1-6): 1  👈 Select Huggingface-Local
```

#### Step 4: Select Model

```
🤖 Local Model Inference (Hugging Face)
==================================================

📋 Popular models:
1. Florence-2 Large (best OCR, ~770M)
   ID: microsoft/Florence-2-large
2. Florence-2 Base (lighter, ~300M)
   ID: microsoft/Florence-2-base
3. Qwen2-VL 7B (strong vision)
   ID: Qwen/Qwen2-VL-7B-Instruct
--------------------------------------------------
0. [Custom Model ID]
==================================================

Select model (1-3) or '0' for custom: 1  👈 Select Florence-2 Large
```

#### Step 5: Watch It Work

```
📥 Loading model: microsoft/Florence-2-large
   Device: CPU
[Model downloads - first time only, ~2-5 minutes]
✅ Model loaded successfully

📄 Processing: electrical_drawing.pdf
📊 Total pages: 195

🚀 Starting OCR/Analysis with microsoft/Florence-2-large...
======================================================================

Processing Pages:   0%|          | 0/195 [00:00<?, ?page/s]
Processing Pages:   1%|          | 1/195 [00:02<07:45,  2.40s/page]
✅ Extracted: ELECTRICAL SCHEMATIC ...
...

✅ OCR complete! Saved to output_ocr_20250127_143022.json
```

#### Step 6: Check Results

```bash
# View the output file
cat output_ocr_20250127_143022.json
```

---

### Tutorial 2: Using Cloud API (Google Gemini - Free Tier)

**Estimated time:** 5 minutes

#### Step 1: Get Free API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API key"
3. Copy the key

#### Step 2: Add to .env File

```bash
nano .env
```

Add or uncomment:
```env
GOOGLE_API_KEY=AIza...your_key_here...
```

Save and exit (Ctrl+X, Y, Enter)

#### Step 3: Run with Gemini

```bash
# Interactive mode
ai-drawing-analyzer document.pdf

# Select: Gemini (option 4)
# Select: Gemini 2.0 Flash (option 1)

# Or command-line
ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp
```

---

### Tutorial 3: Direct Command-Line Usage (Advanced)

#### Extract OCR with Florence-2 Locally

```bash
ai-drawing-analyzer blueprint.pdf -p huggingface-local -m microsoft/Florence-2-large
```

#### Extract with Custom Model

```bash
ai-drawing-analyzer document.pdf -p huggingface-local -m microsoft/Florence-2-base
```

#### Use HuggingFace Router API (Free)

First, get a free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
# Add to .env or use --api-key
ai-drawing-analyzer document.pdf -p huggingface -m Qwen/Qwen2.5-VL-7B-Instruct --api-key hf_xxxxx
```

#### Process Remote PDF

```bash
ai-drawing-analyzer https://example.com/drawing.pdf -p huggingface-local -m microsoft/Florence-2-large
```

---

### Tutorial 4: Batch Processing Multiple PDFs

```bash
# Process multiple PDFs in sequence (model loads once, reused for all)
for pdf in *.pdf; do
  echo "Processing $pdf..."
  ai-drawing-analyzer "$pdf" -p huggingface-local -m microsoft/Florence-2-large
  echo "Done: $pdf"
done

# Combine all results
cat output_ocr_*.json > all_results.jsonl
```

---

### Common Workflows

#### Workflow A: Extract Text from Construction Drawings (Local, Fast)

```bash
# Best for: Technical drawings, blueprints, electrical schematics
# Model: Florence-2-large
# No API key needed

ai-drawing-analyzer construction_drawing.pdf -p huggingface-local -m microsoft/Florence-2-large
```

**Output:** Precise text extraction with OCR task optimized for technical documents

#### Workflow B: Analyze & Understand Content (Cloud API, Smart)

```bash
# Best for: PDFs that need interpretation, general understanding
# Model: Claude 3.5 Sonnet or Gemini
# Requires API key

ai-drawing-analyzer report.pdf -p anthropic -m claude-3-5-sonnet-20241022
```

**Output:** Intelligent analysis with reasoning

#### Workflow C: Quick Testing (Cloud API, Free)

```bash
# Best for: Quick tests, low volume
# Model: Gemini 2.0 Flash
# Free tier: 60 req/minute

ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp
```

**Output:** Fast, free extraction (free tier)

---

## 🔄 JSONL to Text Conversion for LLM Analysis

### Overview

After extracting OCR text from a PDF using the standard workflow, you can convert the JSONL output into a single, comprehensive text file optimized for feeding into Large Language Models like Claude, GPT-4, or other AI assistants.

This feature creates a unified document that preserves the complete drawing context, including:
- Structured metadata from the title block
- Cross-reference index (maps references between pages)
- Legend and symbols reference sections
- Document structure overview
- Complete page-by-page content

### Why Use This Feature?

**Problem:** JSONL format is fragmented (one entry per page), making it difficult for LLMs to understand the full drawing context and relationships between pages.

**Solution:** The text converter creates a coherent document where an LLM can:
- Understand the complete drawing without losing context
- Navigate between related sections via the cross-reference index
- Interpret symbols and legends properly
- Analyze the complete structure in one go

### Usage

#### Method 1: Convert Existing JSONL File

```bash
# Convert a JSONL file to text (standalone)
ai-drawing-analyzer output_ocr_20250127_143022.json

# With custom output filename
ai-drawing-analyzer output_ocr_20250127_143022.json -o my_drawing_complete.txt
```

#### Method 2: Extract PDF and Convert to Text in One Command

```bash
# Extract OCR and convert to text automatically
ai-drawing-analyzer blueprint.pdf -p huggingface-local -m microsoft/Florence-2-large --to-text

# With custom output filename
ai-drawing-analyzer blueprint.pdf -p huggingface-local -m microsoft/Florence-2-large --to-text -o electrical_drawing_complete.txt
```

#### Method 3: Convert Multiple JSONL Files

```bash
# Convert the most recent JSONL file
ai-drawing-analyzer output_ocr_*.json
```

### Output File Structure

The generated text file contains these sections in order:

1. **Header**: Document title and metadata
2. **Document Metadata**: Project info, sheet number, revision, scale, etc.
3. **Cross-Reference Index**: Map of all page references found in the drawing
4. **Legend & Symbols Reference**: All symbols and abbreviations defined in the drawing
5. **Document Structure**: Quick overview of each page's content
6. **Complete Drawing Content**: Full text content for all pages with clear page breaks
7. **Footer**: Summary statistics (total pages, cross-references found, etc.)

### Example Output Structure

```
================================================================================
COMPLETE DRAWING SET - TEXT EXTRACTION FOR LLM ANALYSIS
================================================================================

DOCUMENT METADATA
================================================================================
Project: LEVANDE PROJECT - ELECTRICAL SUBSTATION
Title: Main Distribution Panel - Electrical Schematic
Discipline: Electrical
Sheet: E1.1
Revision: A2
Revision Date: 2025-01-15
Drawn By: Mohammed Shousha
Scale: 1:50
Total Pages: 47
Extracted Date: 2025-01-27T14:30:22Z
================================================================================

CROSS-REFERENCE INDEX
================================================================================
Detail A: pages 5, 12, 47
Panel Schedule: pages 8, 15
Section 3-3: pages 10, 22
Equipment Schedule: pages 18
...
================================================================================

LEGEND & SYMBOLS REFERENCE
================================================================================
Page 1 - Symbols/Legend:
━━━ (solid line): Power Circuit
- - - (dashed line): Control Circuit
⊕: Connection Point
...
================================================================================

DOCUMENT STRUCTURE:
================================================================================
Page 1: Title Block, General Notes, Legend
Page 2: Main Distribution Panel - Front View
Page 3: Main Distribution Panel - Rear View
...
================================================================================

COMPLETE DRAWING CONTENT
================================================================================

────────────────────────────────────────────────────────────────────────────────
PAGE 1 of 47
────────────────────────────────────────────────────────────────────────────────
[Full OCR text from page 1]

────────────────────────────────────────────────────────────────────────────────
PAGE 2 of 47
────────────────────────────────────────────────────────────────────────────────
[Full OCR text from page 2]

... (continues for all pages)
```

### Using the Output with Claude or GPT-4

```python
from anthropic import Anthropic

client = Anthropic()

# Read the generated text file
with open('drawing_complete.txt', 'r') as f:
    drawing_text = f.read()

# Send to Claude for analysis
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": f"""Analyze this complete electrical drawing:

{drawing_text}

Please provide:
1. Summary of the main components
2. Any safety concerns
3. Suggested improvements"""
        }
    ]
)

print(response.content[0].text)
```

### Supported Features

✅ **Metadata Extraction**
- Automatically extracts project name, drawing title, revision, scale, etc. from title block
- Uses regex patterns to find common title block layouts

✅ **Cross-Reference Detection**
- Finds and indexes all page references in the drawing
- Detects patterns like "see page 47", "refer to page X", "detail on page Z"
- Creates a navigable index

✅ **Legend Extraction**
- Automatically identifies legend/symbols pages
- Extracts and includes symbol definitions

✅ **Document Structure**
- Creates overview of all pages with content summaries
- Helps LLMs understand overall drawing organization

✅ **Full Content Preservation**
- Maintains all OCR text content
- Preserves page order and numbers
- Clear page delimiters for readability

### Tips & Best Practices

1. **Metadata Accuracy**: The first page should contain a title block for best results. If metadata isn't extracted, check the OCR quality of page 1.

2. **Reference Detection**: Make sure page references use standard patterns ("page X", "see page X", etc.). Non-standard formats may not be detected.

3. **File Size**: Large drawings (100+ pages) will create large text files. Claude's context window can handle up to ~200k characters depending on model.

4. **OCR Quality**: Better OCR quality = better metadata extraction and reference detection. Use Florence-2-large for best results on complex drawings.

5. **Legend Placement**: Place legend sections on dedicated pages for reliable detection. Keywords like "LEGEND", "SYMBOLS", "ABBREVIATIONS" help identify these pages.

### Advanced Usage

#### Combining with Document Analysis

```bash
# 1. Extract OCR from PDF
ai-drawing-analyzer blueprint.pdf -p huggingface-local -m microsoft/Florence-2-large

# 2. Convert to text
ai-drawing-analyzer output_ocr_20250127_143022.json -o analysis_ready.txt

# 3. Feed to Claude for detailed analysis
cat analysis_ready.txt | pbcopy  # Copy to clipboard (macOS)
# Then paste into Claude Chat
```

#### Batch Processing Multiple PDFs

```bash
#!/bin/bash
# Convert all PDFs in a directory

for pdf in *.pdf; do
    echo "Processing $pdf..."
    ai-drawing-analyzer "$pdf" -p huggingface-local -m microsoft/Florence-2-large --to-text -o "${pdf%.pdf}_complete.txt"
    echo "✅ Created ${pdf%.pdf}_complete.txt"
done
```

### Troubleshooting

**Issue:** Metadata not extracted

**Solution:**
- Check that page 1 has a clear title block
- Verify OCR quality of page 1
- The script tries common patterns; if your title block is unusual, it may not match

**Issue:** Cross-references not detected

**Solution:**
- Check that references use standard patterns like "page X", "see page X"
- Verify the page numbers in OCR text match actual page numbers
- Use consistent formatting for references

**Issue:** Very large output file

**Solution:**
- This is expected for 100+ page drawings
- Consider splitting the PDF into multiple documents
- The text is still optimized for LLM consumption

---

## 📋 Requirements Summary

### Core Dependencies (Always Required)
- `httpx` — HTTP client
- `pymupdf` — PDF processing
- `Pillow` — Image handling

### Optional: Cloud API Support
- `google-auth`, `google-auth-httplib2` — For Gemini API
- `sentencepiece` — Tokenizer (used by some models)
- `numpy` — Numerical operations

### Optional: Local Model Inference
- `transformers` — Model loading framework
- `torch` — Deep learning backend (CPU or GPU)

---

## 🚀 Performance Tips

1. **GPU is 10-100x faster** than CPU for local inference
2. **Smaller models are faster** but less accurate (Florence-2-base vs Florence-2-large)
3. **Batch processing:** Run multiple PDFs in sequence to amortize model loading
4. **API-based models** are better for occasional use; local models for high-volume processing

---

## 📄 License

This project is open-source. See the repository for license details.

---

## 🤝 Contributing

Feel free to submit issues, fork the repo, and create pull requests for improvements!

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review the code comments in `pdf_analyzer_complete.py`
3. Open an issue on GitHub

---

**Last Updated:** January 2025
**Version:** 2.1.0 (with modular architecture and async support)