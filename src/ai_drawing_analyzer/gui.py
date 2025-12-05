"""
AI Drawing Analyzer - Gradio Web Interface

A user-friendly web interface for the AI Drawing Analyzer tool.
Supports single PDF processing, batch processing, and API key management.
"""

import os
import json
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

import gradio as gr

from .clients.factory import ClientFactory
from .processing.pdf import PDFProcessor
from .converters.jsonl_to_text import DrawingTextConverter
from .converters.toon_converter import ToonConverter
from .utils.config import load_env_file, AppConfig
from .utils.files import detect_and_download
from .utils.logging import logger


# ============================================================================
# Configuration & State Management
# ============================================================================

CONFIG_FILE = Path.home() / ".ai_drawing_analyzer" / "config.json"

def load_saved_config() -> dict:
    """Load saved configuration from disk"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(config: dict) -> None:
    """Save configuration to disk"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_api_key(provider: str, saved_config: dict) -> Optional[str]:
    """Get API key for provider from environment or saved config"""
    env_keys = {
        'gemini': 'GOOGLE_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'huggingface': 'HF_TOKEN',
        'openrouter': 'OPENROUTER_API_KEY',
    }

    env_var = env_keys.get(provider)
    if env_var:
        # Check environment first
        if os.getenv(env_var):
            return os.getenv(env_var)
        # Check saved config
        return saved_config.get('api_keys', {}).get(provider)
    return None


# ============================================================================
# Provider & Model Information
# ============================================================================

PROVIDERS = {
    'gemini': {
        'name': 'Google Gemini',
        'description': 'Free tier available, fast, good quality',
        'env_var': 'GOOGLE_API_KEY',
        'models': [
            ('gemini-2.0-flash-exp', 'Gemini 2.0 Flash (Experimental)'),
            ('gemini-1.5-flash', 'Gemini 1.5 Flash'),
            ('gemini-1.5-pro', 'Gemini 1.5 Pro'),
        ],
        'requires_key': True,
    },
    'openai': {
        'name': 'OpenAI',
        'description': 'High quality, paid API',
        'env_var': 'OPENAI_API_KEY',
        'models': [
            ('gpt-4o', 'GPT-4o (Best quality)'),
            ('gpt-4o-mini', 'GPT-4o Mini (Faster, cheaper)'),
        ],
        'requires_key': True,
    },
    'anthropic': {
        'name': 'Anthropic Claude',
        'description': 'High quality, paid API',
        'env_var': 'ANTHROPIC_API_KEY',
        'models': [
            ('claude-3-5-sonnet-20241022', 'Claude 3.5 Sonnet'),
            ('claude-3-5-haiku-20241022', 'Claude 3.5 Haiku'),
        ],
        'requires_key': True,
    },
    'huggingface': {
        'name': 'HuggingFace Router',
        'description': 'Free tier, cloud inference',
        'env_var': 'HF_TOKEN',
        'models': [
            ('Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen2.5-VL 7B'),
            ('mistralai/Pixtral-12B-2409', 'Pixtral 12B'),
        ],
        'requires_key': True,
    },
    'openrouter': {
        'name': 'OpenRouter',
        'description': 'Multi-model gateway',
        'env_var': 'OPENROUTER_API_KEY',
        'models': [
            ('anthropic/claude-3.5-sonnet', 'Claude 3.5 Sonnet'),
            ('openai/gpt-4o', 'GPT-4o'),
            ('google/gemini-pro-vision', 'Gemini Pro Vision'),
        ],
        'requires_key': True,
    },
    'huggingface-local': {
        'name': 'Local Models',
        'description': 'Run models locally (no API key needed)',
        'env_var': None,
        'models': [
            ('microsoft/Florence-2-large', 'Florence-2 Large (Recommended)'),
            ('microsoft/Florence-2-base', 'Florence-2 Base'),
            ('Qwen/Qwen2-VL-7B-Instruct', 'Qwen2-VL 7B'),
        ],
        'requires_key': False,
    },
}


# ============================================================================
# Processing Functions
# ============================================================================

async def process_single_pdf(
    pdf_file,
    provider: str,
    model: str,
    api_key: str,
    to_text: bool,
    to_toon: bool,
    progress=gr.Progress()
) -> Tuple[str, Optional[str], Optional[str], str]:
    """Process a single PDF file"""

    if pdf_file is None:
        return "", None, None, "Please upload a PDF file"

    try:
        # Create client
        if provider == 'huggingface-local':
            client = ClientFactory.create_client(provider, model_id=model)
        else:
            if not api_key:
                return "", None, None, f"API key required for {PROVIDERS[provider]['name']}"
            # Set API key in environment
            env_var = PROVIDERS[provider]['env_var']
            if env_var:
                os.environ[env_var] = api_key
            client = ClientFactory.create_client(provider, api_key)

        # Get config
        config = AppConfig.from_env()

        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="ai_drawing_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_output = os.path.join(output_dir, f"output_{timestamp}.jsonl")

        # Process PDF
        pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        local_path = detect_and_download(pdf_path)

        results = []
        with PDFProcessor(local_path, zoom=config['pdf_zoom_level'], quality=config['jpeg_quality']) as processor:
            page_count = processor.get_page_count()

            for page_num in range(page_count):
                progress((page_num + 1) / page_count, desc=f"Processing page {page_num + 1}/{page_count}")

                try:
                    image_base64 = processor.get_page_as_image_base64(page_num)
                    prompt = "OCR Task: Transcribe all text visible in this image verbatim. Do not summarize or analyze. Output only the text."

                    response = await client.analyze_image(image_base64, prompt, model)

                    result = {
                        "page": page_num + 1,
                        "page_type": "image",
                        "provider": provider,
                        "model": model,
                        "text_content": response,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)

                    with open(jsonl_output, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                except Exception as e:
                    results.append({
                        "page": page_num + 1,
                        "error": str(e)
                    })

        # Convert to text if requested
        text_output = None
        if to_text:
            text_path = jsonl_output.replace('.jsonl', '.txt')
            converter = DrawingTextConverter(jsonl_output)
            converter.convert(text_path)
            text_output = text_path

        # Convert to toon if requested
        toon_output = None
        if to_toon:
            try:
                toon_path = jsonl_output.replace('.jsonl', '.toon')
                toon_converter = ToonConverter()
                toon_converter.convert(jsonl_output, toon_path)
                toon_output = toon_path
            except Exception as e:
                logger.warning(f"Toon conversion failed: {e}")

        # Compile results text
        result_text = f"Processed {page_count} pages successfully!\n\n"
        for r in results:
            if 'error' in r:
                result_text += f"Page {r['page']}: Error - {r['error']}\n"
            else:
                result_text += f"--- Page {r['page']} ---\n{r['text_content']}\n\n"

        return result_text, jsonl_output, text_output, "Processing complete!"

    except Exception as e:
        return "", None, None, f"Error: {str(e)}"


async def process_batch_pdfs_gui(
    folder_path: str,
    provider: str,
    model: str,
    api_key: str,
    recursive: bool,
    to_text: bool,
    to_toon: bool,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Process multiple PDFs from a folder"""

    if not folder_path or not os.path.isdir(folder_path):
        return "", "Please enter a valid folder path"

    try:
        # Find PDF files
        folder = Path(folder_path)
        if recursive:
            pdf_files = sorted(folder.rglob("*.pdf"), key=lambda p: str(p).lower())
        else:
            pdf_files = sorted(folder.glob("*.pdf"), key=lambda p: p.name.lower())

        if not pdf_files:
            return "", f"No PDF files found in {folder_path}"

        # Create client
        if provider == 'huggingface-local':
            client = ClientFactory.create_client(provider, model_id=model)
        else:
            if not api_key:
                return "", f"API key required for {PROVIDERS[provider]['name']}"
            env_var = PROVIDERS[provider]['env_var']
            if env_var:
                os.environ[env_var] = api_key
            client = ClientFactory.create_client(provider, api_key)

        # Create output directory
        output_dir = folder / "batch_output"
        output_dir.mkdir(exist_ok=True)

        config = AppConfig.from_env()
        results_summary = []

        for i, pdf_file in enumerate(pdf_files):
            progress((i + 1) / len(pdf_files), desc=f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})")

            try:
                base_name = pdf_file.stem
                jsonl_output = output_dir / f"{base_name}.jsonl"

                local_path = detect_and_download(str(pdf_file))

                with PDFProcessor(local_path, zoom=config['pdf_zoom_level'], quality=config['jpeg_quality']) as processor:
                    page_count = processor.get_page_count()

                    for page_num in range(page_count):
                        try:
                            image_base64 = processor.get_page_as_image_base64(page_num)
                            prompt = "OCR Task: Transcribe all text visible in this image verbatim."

                            response = await client.analyze_image(image_base64, prompt, model)

                            result = {
                                "page": page_num + 1,
                                "provider": provider,
                                "model": model,
                                "text_content": response,
                                "timestamp": datetime.now().isoformat()
                            }

                            with open(jsonl_output, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(result, ensure_ascii=False) + '\n')

                        except Exception as e:
                            logger.error(f"Error on page {page_num + 1}: {e}")

                # Convert if requested
                if to_text:
                    text_path = output_dir / f"{base_name}.txt"
                    converter = DrawingTextConverter(str(jsonl_output))
                    converter.convert(str(text_path))

                if to_toon:
                    try:
                        toon_path = output_dir / f"{base_name}.toon"
                        toon_converter = ToonConverter()
                        toon_converter.convert(str(jsonl_output), str(toon_path))
                    except Exception:
                        pass

                results_summary.append(f"[OK] {pdf_file.name} - {page_count} pages")

            except Exception as e:
                results_summary.append(f"[FAILED] {pdf_file.name} - {str(e)}")

        # Save summary
        summary_file = output_dir / "_batch_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("\n".join(results_summary))

        return "\n".join(results_summary), f"Batch processing complete! Output saved to: {output_dir}"

    except Exception as e:
        return "", f"Error: {str(e)}"


# ============================================================================
# UI Helper Functions
# ============================================================================

def update_models(provider: str) -> gr.Dropdown:
    """Update model dropdown based on selected provider"""
    if provider in PROVIDERS:
        models = PROVIDERS[provider]['models']
        choices = [(name, id) for id, name in models]
        return gr.Dropdown(choices=choices, value=models[0][0] if models else None)
    return gr.Dropdown(choices=[], value=None)


def update_api_key_visibility(provider: str) -> gr.Textbox:
    """Update API key field visibility based on provider"""
    if provider in PROVIDERS:
        requires_key = PROVIDERS[provider]['requires_key']
        return gr.Textbox(visible=requires_key)
    return gr.Textbox(visible=True)


def save_api_keys(gemini_key, openai_key, anthropic_key, hf_key, openrouter_key) -> str:
    """Save API keys to config file"""
    config = load_saved_config()
    config['api_keys'] = {
        'gemini': gemini_key or '',
        'openai': openai_key or '',
        'anthropic': anthropic_key or '',
        'huggingface': hf_key or '',
        'openrouter': openrouter_key or '',
    }
    save_config(config)

    # Also set in environment
    if gemini_key:
        os.environ['GOOGLE_API_KEY'] = gemini_key
    if openai_key:
        os.environ['OPENAI_API_KEY'] = openai_key
    if anthropic_key:
        os.environ['ANTHROPIC_API_KEY'] = anthropic_key
    if hf_key:
        os.environ['HF_TOKEN'] = hf_key
    if openrouter_key:
        os.environ['OPENROUTER_API_KEY'] = openrouter_key

    return "API keys saved successfully!"


def load_api_keys() -> Tuple[str, str, str, str, str]:
    """Load saved API keys"""
    config = load_saved_config()
    keys = config.get('api_keys', {})
    return (
        keys.get('gemini', '') or os.getenv('GOOGLE_API_KEY', ''),
        keys.get('openai', '') or os.getenv('OPENAI_API_KEY', ''),
        keys.get('anthropic', '') or os.getenv('ANTHROPIC_API_KEY', ''),
        keys.get('huggingface', '') or os.getenv('HF_TOKEN', ''),
        keys.get('openrouter', '') or os.getenv('OPENROUTER_API_KEY', ''),
    )


def get_resource_info() -> str:
    """Get system resource information"""
    try:
        return AppConfig.get_resource_info()
    except Exception:
        return "Could not detect system resources"


# ============================================================================
# Main Gradio Interface
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create the main Gradio interface"""

    # Load saved config
    saved_config = load_saved_config()
    saved_keys = saved_config.get('api_keys', {})

    with gr.Blocks(
        title="AI Drawing Analyzer",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .status-box { padding: 10px; border-radius: 5px; }
        """
    ) as demo:

        gr.Markdown(
            """
            # AI Drawing Analyzer
            ### Extract text from PDFs using Vision-Language Models
            """,
            elem_classes=["main-header"]
        )

        with gr.Tabs():
            # ================================================================
            # Tab 1: Single PDF Processing
            # ================================================================
            with gr.TabItem("Single PDF"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_input = gr.File(
                            label="Upload PDF",
                            file_types=[".pdf"],
                            type="filepath"
                        )

                        provider_single = gr.Dropdown(
                            choices=[(v['name'], k) for k, v in PROVIDERS.items()],
                            value='gemini',
                            label="AI Provider"
                        )

                        model_single = gr.Dropdown(
                            choices=[(name, id) for id, name in PROVIDERS['gemini']['models']],
                            value='gemini-2.0-flash-exp',
                            label="Model"
                        )

                        api_key_single = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Enter API key (or save in Settings)",
                            value=saved_keys.get('gemini', '') or os.getenv('GOOGLE_API_KEY', '')
                        )

                        with gr.Row():
                            to_text_single = gr.Checkbox(label="Convert to Text", value=True)
                            to_toon_single = gr.Checkbox(label="Convert to Toon", value=False)

                        process_btn = gr.Button("Process PDF", variant="primary")

                    with gr.Column(scale=2):
                        output_text = gr.Textbox(
                            label="OCR Results",
                            lines=20,
                            max_lines=30
                        )

                        with gr.Row():
                            jsonl_output = gr.File(label="JSONL Output")
                            text_output = gr.File(label="Text Output")

                        status_single = gr.Textbox(label="Status", interactive=False)

                # Event handlers
                provider_single.change(
                    fn=lambda p: (
                        gr.Dropdown(choices=[(name, id) for id, name in PROVIDERS[p]['models']],
                                   value=PROVIDERS[p]['models'][0][0] if PROVIDERS[p]['models'] else None),
                        gr.Textbox(visible=PROVIDERS[p]['requires_key'],
                                  value=saved_keys.get(p, '') or os.getenv(PROVIDERS[p].get('env_var', ''), ''))
                    ),
                    inputs=[provider_single],
                    outputs=[model_single, api_key_single]
                )

                process_btn.click(
                    fn=lambda *args: asyncio.run(process_single_pdf(*args)),
                    inputs=[pdf_input, provider_single, model_single, api_key_single,
                           to_text_single, to_toon_single],
                    outputs=[output_text, jsonl_output, text_output, status_single]
                )

            # ================================================================
            # Tab 2: Batch Processing
            # ================================================================
            with gr.TabItem("Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        folder_input = gr.Textbox(
                            label="Folder Path",
                            placeholder="/path/to/pdf/folder"
                        )

                        recursive_check = gr.Checkbox(
                            label="Include subfolders",
                            value=False
                        )

                        provider_batch = gr.Dropdown(
                            choices=[(v['name'], k) for k, v in PROVIDERS.items()],
                            value='gemini',
                            label="AI Provider"
                        )

                        model_batch = gr.Dropdown(
                            choices=[(name, id) for id, name in PROVIDERS['gemini']['models']],
                            value='gemini-2.0-flash-exp',
                            label="Model"
                        )

                        api_key_batch = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Enter API key",
                            value=saved_keys.get('gemini', '') or os.getenv('GOOGLE_API_KEY', '')
                        )

                        with gr.Row():
                            to_text_batch = gr.Checkbox(label="Convert to Text", value=True)
                            to_toon_batch = gr.Checkbox(label="Convert to Toon", value=False)

                        batch_btn = gr.Button("Start Batch Processing", variant="primary")

                    with gr.Column(scale=2):
                        batch_results = gr.Textbox(
                            label="Processing Results",
                            lines=20,
                            max_lines=30
                        )
                        status_batch = gr.Textbox(label="Status", interactive=False)

                # Event handlers
                provider_batch.change(
                    fn=lambda p: (
                        gr.Dropdown(choices=[(name, id) for id, name in PROVIDERS[p]['models']],
                                   value=PROVIDERS[p]['models'][0][0] if PROVIDERS[p]['models'] else None),
                        gr.Textbox(visible=PROVIDERS[p]['requires_key'],
                                  value=saved_keys.get(p, '') or os.getenv(PROVIDERS[p].get('env_var', ''), ''))
                    ),
                    inputs=[provider_batch],
                    outputs=[model_batch, api_key_batch]
                )

                batch_btn.click(
                    fn=lambda *args: asyncio.run(process_batch_pdfs_gui(*args)),
                    inputs=[folder_input, provider_batch, model_batch, api_key_batch,
                           recursive_check, to_text_batch, to_toon_batch],
                    outputs=[batch_results, status_batch]
                )

            # ================================================================
            # Tab 3: API Keys Settings
            # ================================================================
            with gr.TabItem("Settings"):
                gr.Markdown("### API Key Management")
                gr.Markdown("Enter your API keys below. They will be saved locally for future use.")

                with gr.Group():
                    gemini_key = gr.Textbox(
                        label="Google Gemini API Key",
                        type="password",
                        placeholder="Get from: https://makersuite.google.com/app/apikey",
                        value=saved_keys.get('gemini', '') or os.getenv('GOOGLE_API_KEY', '')
                    )

                    openai_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="Get from: https://platform.openai.com/api-keys",
                        value=saved_keys.get('openai', '') or os.getenv('OPENAI_API_KEY', '')
                    )

                    anthropic_key = gr.Textbox(
                        label="Anthropic API Key",
                        type="password",
                        placeholder="Get from: https://console.anthropic.com/",
                        value=saved_keys.get('anthropic', '') or os.getenv('ANTHROPIC_API_KEY', '')
                    )

                    hf_key = gr.Textbox(
                        label="HuggingFace Token",
                        type="password",
                        placeholder="Get from: https://huggingface.co/settings/tokens",
                        value=saved_keys.get('huggingface', '') or os.getenv('HF_TOKEN', '')
                    )

                    openrouter_key = gr.Textbox(
                        label="OpenRouter API Key",
                        type="password",
                        placeholder="Get from: https://openrouter.ai/keys",
                        value=saved_keys.get('openrouter', '') or os.getenv('OPENROUTER_API_KEY', '')
                    )

                save_keys_btn = gr.Button("Save API Keys", variant="primary")
                keys_status = gr.Textbox(label="Status", interactive=False)

                save_keys_btn.click(
                    fn=save_api_keys,
                    inputs=[gemini_key, openai_key, anthropic_key, hf_key, openrouter_key],
                    outputs=[keys_status]
                )

                gr.Markdown("---")
                gr.Markdown("### System Information")

                resource_info = gr.Textbox(
                    label="Detected Resources",
                    value=get_resource_info(),
                    interactive=False
                )

                refresh_btn = gr.Button("Refresh")
                refresh_btn.click(fn=get_resource_info, outputs=[resource_info])

            # ================================================================
            # Tab 4: Help
            # ================================================================
            with gr.TabItem("Help"):
                gr.Markdown("""
                ## Quick Start Guide

                ### 1. Set Up API Keys
                - Go to the **Settings** tab
                - Enter your API keys for the providers you want to use
                - Click **Save API Keys**

                ### 2. Process a Single PDF
                - Go to the **Single PDF** tab
                - Upload your PDF file
                - Select a provider and model
                - Click **Process PDF**

                ### 3. Batch Process Multiple PDFs
                - Go to the **Batch Processing** tab
                - Enter the folder path containing your PDFs
                - Select a provider and model
                - Click **Start Batch Processing**

                ---

                ## Provider Comparison

                | Provider | Free Tier | Speed | Quality | Best For |
                |----------|-----------|-------|---------|----------|
                | **Gemini** | Yes (60 req/min) | Fast | Good | General use |
                | **OpenAI** | No | Fast | Excellent | High accuracy |
                | **Claude** | No | Medium | Excellent | Complex documents |
                | **Local Models** | Yes (Free) | Slow* | Good | Privacy, offline |

                *Local models require GPU for fast processing

                ---

                ## Tips

                - **For best results**: Use Gemini 2.0 Flash or GPT-4o
                - **For free processing**: Use Gemini (free tier) or Local Models
                - **For privacy**: Use Local Models (no data sent to cloud)
                - **For batch jobs**: Enable "Resume" to continue if interrupted

                ---

                ## Troubleshooting

                - **API Key errors**: Check your key in Settings tab
                - **Slow processing**: Try a faster model or reduce PDF quality
                - **Memory errors**: Use lower zoom level or smaller PDFs
                """)

        gr.Markdown(
            """
            ---
            <center>AI Drawing Analyzer |
            <a href="https://github.com/mohandshamada/AI_Drawings_Reader">GitHub</a></center>
            """,
            elem_classes=["footer"]
        )

    return demo


def main():
    """Launch the Gradio interface"""
    # Load environment variables
    load_env_file()

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to get a public URL
        show_error=True
    )


if __name__ == "__main__":
    main()
