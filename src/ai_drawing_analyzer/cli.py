import argparse
import sys
import os
import json
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional, Set, List
from tqdm import tqdm

from .clients.factory import ClientFactory
from .processing.pdf import PDFProcessor
from .converters.jsonl_to_text import DrawingTextConverter
from .converters.toon_converter import ToonConverter
from .utils.config import load_env_file, AppConfig
from .utils.files import detect_and_download
from .utils.logging import logger


def find_pdf_files(folder_path: str) -> List[Path]:
    """Find all PDF files in a folder (non-recursive by default)"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    pdf_files = sorted(folder.glob("*.pdf"), key=lambda p: p.name.lower())
    return pdf_files


def find_pdf_files_recursive(folder_path: str) -> List[Path]:
    """Find all PDF files in a folder and subfolders"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    pdf_files = sorted(folder.rglob("*.pdf"), key=lambda p: str(p).lower())
    return pdf_files

def get_processed_pages(output_file: str) -> Set[int]:
    """Get set of already processed pages from output file"""
    processed = set()
    if not os.path.exists(output_file):
        return processed

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'page' in data:
                        processed.add(data['page'])
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logger.warning(f"Could not read processed pages: {e}")

    return processed

async def process_pdf_async(pdf_path: str, client, model: str, provider: str, resume: bool = False, output_file: Optional[str] = None):
    """Process PDF with optional resume capability"""
    logger.info(f"Processing: {pdf_path}")
    local_path = detect_and_download(pdf_path)

    config = AppConfig.from_env()

    with PDFProcessor(local_path, zoom=config['pdf_zoom_level'], quality=config['jpeg_quality']) as processor:
        page_count = processor.get_page_count()
        logger.info(f"Total pages: {page_count}")

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output_ocr_{timestamp}.jsonl"

        # Check for resume
        processed_pages = set()
        if resume and os.path.exists(output_file):
            processed_pages = get_processed_pages(output_file)
            logger.info(f"Resuming: {len(processed_pages)} pages already processed")

        logger.info(f"Starting OCR/Analysis with {model}...")

        pages_to_process = [p for p in range(page_count) if (p + 1) not in processed_pages]

        # We process pages sequentially to respect rate limits and memory,
        # but the architecture supports parallelizing this in future (with semaphores)
        for page_num in tqdm(pages_to_process, desc="Processing Pages", unit="page"):
            try:
                # 1. Convert PDF Page to Image
                image_base64 = processor.get_page_as_image_base64(page_num)

                # 2. Define OCR Prompt
                prompt = "OCR Task: Transcribe all text visible in this image verbatim. Do not summarize or analyze. Output only the text."

                # 3. Analyze
                response = await client.analyze_image(image_base64, prompt, model)

                result = {
                    "page": page_num + 1,
                    "page_type": "image",
                    "provider": provider,
                    "model": model,
                    "text_content": response,
                    "timestamp": datetime.now().isoformat()
                }

                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                logger.error(f"Error on page {page_num + 1}: {e}")
                # Continue to next page

        logger.info(f"OCR complete! Saved to {output_file}")
        return output_file


async def process_batch_pdfs(
    pdf_files: List[Path],
    client,
    model: str,
    provider: str,
    output_dir: str,
    resume: bool = False,
    to_text: bool = False,
    to_toon: bool = False
) -> dict:
    """Process multiple PDF files from a folder

    Args:
        pdf_files: List of PDF file paths to process
        client: AI client instance
        model: Model ID to use
        provider: Provider name
        output_dir: Directory to save output files
        resume: Whether to resume interrupted processing
        to_text: Convert JSONL to text format
        to_toon: Convert to Toon format

    Returns:
        Dictionary with processing results summary
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "total": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "files": []
    }

    logger.info(f"Starting batch processing of {len(pdf_files)} PDF files")
    logger.info(f"Output directory: {output_dir}")

    # Process each PDF with overall progress bar
    for i, pdf_file in enumerate(tqdm(pdf_files, desc="Processing PDFs", unit="file")):
        file_result = {
            "file": str(pdf_file),
            "filename": pdf_file.name,
            "status": "pending",
            "output_jsonl": None,
            "output_text": None,
            "output_toon": None,
            "error": None
        }

        # Generate output filenames based on PDF name
        base_name = pdf_file.stem  # filename without extension
        jsonl_output = output_path / f"{base_name}.jsonl"
        text_output = output_path / f"{base_name}.txt"
        toon_output = output_path / f"{base_name}.toon"

        # Check if already fully processed (for resume)
        if resume and jsonl_output.exists():
            # Check if processing was complete by verifying file has content
            try:
                with open(jsonl_output, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if lines:  # Has some content, may be partial
                    logger.info(f"[{i+1}/{len(pdf_files)}] Resuming: {pdf_file.name}")
            except Exception:
                pass

        try:
            logger.info(f"[{i+1}/{len(pdf_files)}] Processing: {pdf_file.name}")

            # Process the PDF
            output_file = await process_pdf_async(
                str(pdf_file),
                client,
                model,
                provider,
                resume=resume,
                output_file=str(jsonl_output)
            )

            file_result["output_jsonl"] = str(jsonl_output)
            file_result["status"] = "success"

            # Convert to text if requested
            if to_text and output_file:
                try:
                    converter = DrawingTextConverter(output_file)
                    converter.convert(str(text_output))
                    file_result["output_text"] = str(text_output)
                    logger.info(f"  -> Text output: {text_output.name}")
                except Exception as e:
                    logger.warning(f"  -> Text conversion failed: {e}")

            # Convert to Toon if requested
            if to_toon and output_file:
                try:
                    toon_converter = ToonConverter()
                    toon_converter.convert(output_file, str(toon_output))
                    file_result["output_toon"] = str(toon_output)
                    logger.info(f"  -> Toon output: {toon_output.name}")
                except RuntimeError as e:
                    logger.warning(f"  -> Toon conversion skipped: {e}")
                except Exception as e:
                    logger.warning(f"  -> Toon conversion failed: {e}")

            results["successful"] += 1

        except Exception as e:
            logger.error(f"[{i+1}/{len(pdf_files)}] Failed: {pdf_file.name} - {e}")
            file_result["status"] = "failed"
            file_result["error"] = str(e)
            results["failed"] += 1

        results["files"].append(file_result)

    # Save batch results summary
    summary_file = output_path / "_batch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nBatch processing complete!")
    logger.info(f"  Total: {results['total']}")
    logger.info(f"  Successful: {results['successful']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Summary saved to: {summary_file}")

    return results


async def interactive_provider_selection():
    print("\n🤖 Available AI Providers:")
    print("=" * 70)
    providers = list(ClientFactory.PROVIDERS.keys())
    
    for i, provider in enumerate(providers, 1):
        print(f"{i}. {provider.title()}")
    print("=" * 70)
    
    while True:
        try:
            choice = input(f"\nSelect provider (1-{len(providers)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(providers):
                return providers[idx]
            print("Invalid choice.")
        except ValueError:
            pass

async def interactive_model_selection(client, provider: str) -> Optional[str]:
    if provider == 'huggingface-local':
        # Quick list for local
        models = [
            {"id": "microsoft/Florence-2-large", "name": "Florence-2 Large"},
            {"id": "microsoft/Florence-2-base", "name": "Florence-2 Base"},
            {"id": "Qwen/Qwen2-VL-7B-Instruct", "name": "Qwen2-VL 7B"},
        ]
    else:
        try:
            print("Fetching available models...")
            models = await client.get_available_models()
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return None

    print(f"\n📋 Available Models for {provider}:")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m['name']} ({m['id']})")
    
    if provider in ['huggingface', 'huggingface-local']:
        print("0. [Custom Model ID]")

    while True:
        choice = input("Select model (number) or 'q' to quit: ").strip()
        if choice.lower() == 'q': return None
        
        if choice == '0' and provider in ['huggingface', 'huggingface-local']:
            return input("Enter Model ID: ").strip()
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]['id']
        except ValueError:
            pass

def main():
    parser = argparse.ArgumentParser(
        description="AI Drawing Analyzer: Extract text from PDFs using Vision Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  ai-drawing-analyzer document.pdf

  # Specify provider and model
  ai-drawing-analyzer document.pdf -p openai -m gpt-4o

  # Resume interrupted processing
  ai-drawing-analyzer document.pdf --resume -o output.jsonl

  # Export to multiple formats
  ai-drawing-analyzer document.pdf -p gemini -m gemini-2.0-flash-exp --to-text --to-toon

  # Convert JSONL to text
  ai-drawing-analyzer output.jsonl --to-text --output-text document.txt

  # Export to Toon format (requires Node.js: pnpm install)
  ai-drawing-analyzer document.pdf -p huggingface-local -m microsoft/Florence-2-large --to-toon

  # Batch process all PDFs in a folder
  ai-drawing-analyzer --folder /path/to/pdfs -p gemini -m gemini-2.0-flash-exp --output-dir ./output

  # Batch process with text and toon conversion
  ai-drawing-analyzer -f ./documents -p openai -m gpt-4o --output-dir ./results --to-text --to-toon

  # Batch process recursively (including subfolders)
  ai-drawing-analyzer --folder ./pdfs --recursive -p gemini -m gemini-2.0-flash-exp --output-dir ./output
        """
    )
    parser.add_argument('pdf', nargs='?', help='PDF file path, URL, or JSONL file for conversion')
    parser.add_argument('--folder', '-f', help='Folder containing PDF files for batch processing')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search for PDFs recursively in subfolders (use with --folder)')
    parser.add_argument('--output-dir', '-d', default='./batch_output',
                        help='Output directory for batch processing (default: ./batch_output)')
    parser.add_argument('--provider', '-p', choices=list(ClientFactory.PROVIDERS.keys()),
                        help='AI provider (default: interactive)')
    parser.add_argument('--model', '-m', help='Model ID (default: interactive selection)')
    parser.add_argument('--api-key', '-k', help='API key (default: from environment)')
    parser.add_argument('--env', '-e', default='.env', help='Path to .env file (default: .env)')
    parser.add_argument('--output', '-o', help='Output JSONL file path (default: auto-generated)')
    parser.add_argument('--resume', action='store_true', help='Resume processing from last completed page')
    parser.add_argument('--to-text', action='store_true', help='Convert JSONL to formatted text')
    parser.add_argument('--output-text', default='drawing_complete.txt',
                        help='Output text file path for --to-text (default: drawing_complete.txt)')
    parser.add_argument('--to-toon', action='store_true', help='Convert output to Toon format')
    parser.add_argument('--output-toon', help='Output Toon file path (default: auto-generated)')

    args = parser.parse_args()

    # Validate arguments
    if not args.folder and not args.pdf:
        parser.error("Either a PDF file or --folder must be specified")

    try:
        # JSONL Conversion Mode (single file only)
        if args.pdf and (args.pdf.endswith('.json') or args.pdf.endswith('.jsonl')):
            logger.info(f"Converting JSONL to text: {args.pdf}")
            converter = DrawingTextConverter(args.pdf)
            converter.convert(args.output_text)
            logger.info(f"Conversion complete! Output: {args.output_text}")
            return

        # Config Setup
        env_vars = load_env_file(args.env)

        # Batch Processing Mode
        if args.folder:
            async def run_batch():
                provider = args.provider
                if not provider:
                    provider = await interactive_provider_selection()
                if not provider:
                    logger.error("No provider selected")
                    return

                try:
                    model = args.model
                    client = None

                    if provider == 'huggingface-local':
                        if not model:
                            model = await interactive_model_selection(None, provider)
                        if not model:
                            logger.info("Model selection cancelled")
                            return
                        client = ClientFactory.create_client(provider, model_id=model)
                    else:
                        # API Clients
                        api_key = args.api_key
                        client = ClientFactory.create_client(provider, api_key)
                        if not model:
                            model = await interactive_model_selection(client, provider)
                        if not model:
                            logger.info("Model selection cancelled")
                            return

                    # Find PDF files
                    if args.recursive:
                        pdf_files = find_pdf_files_recursive(args.folder)
                    else:
                        pdf_files = find_pdf_files(args.folder)

                    if not pdf_files:
                        logger.error(f"No PDF files found in: {args.folder}")
                        return

                    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")

                    # Run batch processing
                    results = await process_batch_pdfs(
                        pdf_files=pdf_files,
                        client=client,
                        model=model,
                        provider=provider,
                        output_dir=args.output_dir,
                        resume=args.resume,
                        to_text=args.to_text,
                        to_toon=args.to_toon
                    )

                    # Print final summary
                    print(f"\n{'='*60}")
                    print(f"BATCH PROCESSING COMPLETE")
                    print(f"{'='*60}")
                    print(f"  Total files:  {results['total']}")
                    print(f"  Successful:   {results['successful']}")
                    print(f"  Failed:       {results['failed']}")
                    print(f"  Output dir:   {args.output_dir}")
                    print(f"{'='*60}")

                except KeyError as e:
                    logger.error(f"Missing API key: {e}. Set the environment variable or use --api-key")
                except ValueError as e:
                    logger.error(f"Configuration error: {e}")
                except httpx.HTTPError as e:
                    logger.error(f"Network error: {e}")
                except httpx.TimeoutException as e:
                    logger.error(f"Request timeout: {e}")
                except FileNotFoundError as e:
                    logger.error(f"File/Folder not found: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    if os.getenv("DEBUG"):
                        import traceback
                        traceback.print_exc()

            asyncio.run(run_batch())
            return

        # Single File Processing Mode
        async def run_app():
            provider = args.provider
            if not provider:
                provider = await interactive_provider_selection()
            if not provider:
                logger.error("No provider selected")
                return

            try:
                model = args.model
                client = None

                if provider == 'huggingface-local':
                    if not model:
                        model = await interactive_model_selection(None, provider)
                    if not model:
                        logger.info("Model selection cancelled")
                        return
                    client = ClientFactory.create_client(provider, model_id=model)
                else:
                    # API Clients
                    api_key = args.api_key
                    client = ClientFactory.create_client(provider, api_key)
                    if not model:
                        model = await interactive_model_selection(client, provider)
                    if not model:
                        logger.info("Model selection cancelled")
                        return

                output_file = await process_pdf_async(
                    args.pdf, client, model, provider,
                    resume=args.resume,
                    output_file=args.output
                )

                if args.to_text and output_file:
                    logger.info(f"Converting output to text: {args.output_text}")
                    converter = DrawingTextConverter(output_file)
                    converter.convert(args.output_text)

                if args.to_toon and output_file:
                    try:
                        toon_output = args.output_toon or output_file.replace('.jsonl', '.toon')
                        logger.info(f"Converting output to Toon format: {toon_output}")
                        toon_converter = ToonConverter()
                        toon_converter.convert(output_file, toon_output)
                        logger.info(f"Toon format export complete: {toon_output}")
                    except RuntimeError as e:
                        logger.warning(f"Toon conversion skipped: {e}")
                        logger.warning(f"   Install Node.js dependencies: pnpm install")
                    except Exception as e:
                        logger.error(f"Toon conversion failed: {e}")

            except KeyError as e:
                logger.error(f"Missing API key: {e}. Set the environment variable or use --api-key")
            except ValueError as e:
                logger.error(f"Configuration error: {e}")
            except httpx.HTTPError as e:
                logger.error(f"Network error: {e}")
            except httpx.TimeoutException as e:
                logger.error(f"Request timeout: {e}")
            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if os.getenv("DEBUG"):
                    import traceback
                    traceback.print_exc()

        asyncio.run(run_app())

    except KeyboardInterrupt:
        logger.info("Processing cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
