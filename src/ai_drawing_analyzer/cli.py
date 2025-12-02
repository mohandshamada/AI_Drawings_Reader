import argparse
import sys
import os
import json
import asyncio
import httpx
from datetime import datetime
from typing import Optional, Set
from tqdm import tqdm

from .clients.factory import ClientFactory
from .processing.pdf import PDFProcessor
from .converters.jsonl_to_text import DrawingTextConverter
from .converters.toon_converter import ToonConverter
from .converters.ifc_converter import IFCConverter
from .converters.dwg_converter import DWGConverter
from .utils.config import load_env_file, AppConfig
from .utils.files import detect_and_download
from .utils.logging import logger


def get_file_type(file_path: str) -> str:
    """Determine file type from extension"""
    ext = os.path.splitext(file_path.lower())[1]
    if ext == '.pdf':
        return 'pdf'
    elif ext == '.ifc':
        return 'ifc'
    elif ext in ('.dwg', '.dxf'):
        return 'cad'
    elif ext in ('.json', '.jsonl'):
        return 'jsonl'
    else:
        return 'unknown'

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
        description="AI Drawing Analyzer: Extract text from PDFs, analyze IFC/DWG files using Vision Language Models",
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

  # Convert IFC file to JSON (BIM analysis with clash detection)
  ai-drawing-analyzer model.ifc --to-json

  # Analyze IFC for LLM with clash detection
  ai-drawing-analyzer model.ifc --analyze

  # Convert DWG/DXF file to JSON (CAD analysis)
  ai-drawing-analyzer drawing.dwg --to-json

  # Convert IFC to Toon format
  ai-drawing-analyzer model.ifc --to-toon
        """
    )
    parser.add_argument('input', help='Input file: PDF, IFC, DWG, DXF, or JSONL')
    parser.add_argument('--provider', '-p', choices=list(ClientFactory.PROVIDERS.keys()),
                        help='AI provider (default: interactive)')
    parser.add_argument('--model', '-m', help='Model ID (default: interactive selection)')
    parser.add_argument('--api-key', '-k', help='API key (default: from environment)')
    parser.add_argument('--env', '-e', default='.env', help='Path to .env file (default: .env)')
    parser.add_argument('--output', '-o', help='Output file path (default: auto-generated)')
    parser.add_argument('--resume', action='store_true', help='Resume processing from last completed page')
    parser.add_argument('--to-text', action='store_true', help='Convert JSONL to formatted text')
    parser.add_argument('--output-text', default='drawing_complete.txt',
                        help='Output text file path for --to-text (default: drawing_complete.txt)')
    parser.add_argument('--to-toon', action='store_true', help='Convert output to Toon format')
    parser.add_argument('--output-toon', help='Output Toon file path (default: auto-generated)')

    # IFC/DWG specific options
    parser.add_argument('--to-json', action='store_true',
                        help='Convert IFC/DWG to JSON format (for LLM analysis)')
    parser.add_argument('--output-json', help='Output JSON file path')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze IFC/DWG and output LLM-friendly summary')
    parser.add_argument('--detect-clashes', action='store_true', default=True,
                        help='Detect geometric clashes in IFC files (default: True)')
    parser.add_argument('--no-clashes', action='store_true',
                        help='Disable clash detection for faster processing')
    parser.add_argument('--detect-issues', action='store_true', default=True,
                        help='Detect modeling issues (default: True)')
    parser.add_argument('--no-issues', action='store_true',
                        help='Disable issue detection')
    parser.add_argument('--include-geometry', action='store_true', default=True,
                        help='Include geometry/bounding boxes in IFC output (default: True)')
    parser.add_argument('--no-geometry', action='store_true',
                        help='Exclude geometry for faster processing')
    parser.add_argument('--include-entities', action='store_true', default=True,
                        help='Include all entities in CAD output (default: True)')
    parser.add_argument('--no-entities', action='store_true',
                        help='Exclude entities for smaller output')
    parser.add_argument('--oda-converter', help='Path to ODA File Converter for DWG support')

    args = parser.parse_args()

    try:
        # Determine file type
        file_type = get_file_type(args.input)

        # Handle negation flags
        detect_clashes = not args.no_clashes
        detect_issues = not args.no_issues
        include_geometry = not args.no_geometry
        include_entities = not args.no_entities

        # IFC File Processing
        if file_type == 'ifc':
            logger.info(f"Processing IFC file: {args.input}")
            ifc_converter = IFCConverter()

            if not ifc_converter.is_available():
                logger.error("IFC processing requires ifcopenshell. Install with: pip install ifcopenshell")
                sys.exit(1)

            if args.analyze:
                # LLM-focused analysis output
                result = ifc_converter.analyze_for_llm(args.input, include_elements=include_entities)
                output_path = args.output_json or args.input.replace('.ifc', '_analysis.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info(f"Analysis complete! Output: {output_path}")

                # Print summary
                print(f"\n{'='*60}")
                print("IFC ANALYSIS SUMMARY")
                print(f"{'='*60}")
                print(f"Project: {result['project_summary']['name']}")
                print(f"Elements: {result['project_summary']['total_elements']}")
                print(f"Clashes: {result['clash_analysis']['total_clashes']}")
                print(f"  - Critical: {result['clash_analysis']['by_severity'].get('critical', 0)}")
                print(f"  - Major: {result['clash_analysis']['by_severity'].get('major', 0)}")
                print(f"Issues: {result['issue_analysis']['total_issues']}")
                print(f"\nRecommendations:")
                for rec in result['recommendations']:
                    print(f"  - {rec}")
                print(f"{'='*60}\n")
                return

            if args.to_json:
                output_path = args.output_json or args.input.replace('.ifc', '.json')
                ifc_converter.convert_to_json(
                    args.input, output_path,
                    include_geometry=include_geometry,
                    detect_clashes=detect_clashes,
                    detect_issues=detect_issues
                )
                return

            if args.to_toon:
                output_path = args.output_toon or args.input.replace('.ifc', '.toon')
                ifc_converter.convert_to_toon(
                    args.input, output_path,
                    include_geometry=include_geometry,
                    detect_clashes=detect_clashes,
                    detect_issues=detect_issues
                )
                return

            # Default: convert to JSON
            output_path = args.output_json or args.output or args.input.replace('.ifc', '.json')
            ifc_converter.convert_to_json(
                args.input, output_path,
                include_geometry=include_geometry,
                detect_clashes=detect_clashes,
                detect_issues=detect_issues
            )
            return

        # DWG/DXF File Processing
        if file_type == 'cad':
            logger.info(f"Processing CAD file: {args.input}")
            dwg_converter = DWGConverter(oda_converter_path=args.oda_converter)

            if not dwg_converter.is_available():
                logger.error("CAD processing requires ezdxf. Install with: pip install ezdxf")
                sys.exit(1)

            if args.analyze:
                # LLM-focused analysis output
                result = dwg_converter.analyze_for_llm(args.input, include_entities=include_entities)
                output_path = args.output_json or args.input.rsplit('.', 1)[0] + '_analysis.json'
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info(f"Analysis complete! Output: {output_path}")

                # Print summary
                print(f"\n{'='*60}")
                print("CAD ANALYSIS SUMMARY")
                print(f"{'='*60}")
                print(f"File: {result['drawing_summary']['file_name']}")
                print(f"Format: {result['drawing_summary']['file_format']}")
                print(f"Entities: {result['drawing_summary']['total_entities']}")
                print(f"Layers: {result['layer_analysis']['total_layers']}")
                print(f"Blocks: {result['block_analysis']['total_blocks']}")
                print(f"Issues: {result['issue_analysis']['total_issues']}")
                print(f"\nRecommendations:")
                for rec in result['recommendations']:
                    print(f"  - {rec}")
                print(f"{'='*60}\n")
                return

            if args.to_json:
                output_path = args.output_json or args.input.rsplit('.', 1)[0] + '.json'
                dwg_converter.convert_to_json(
                    args.input, output_path,
                    include_entities=include_entities,
                    include_text_content=True,
                    detect_issues=detect_issues
                )
                return

            if args.to_toon:
                output_path = args.output_toon or args.input.rsplit('.', 1)[0] + '.toon'
                dwg_converter.convert_to_toon(
                    args.input, output_path,
                    include_entities=include_entities,
                    include_text_content=True,
                    detect_issues=detect_issues
                )
                return

            # Default: convert to JSON
            output_path = args.output_json or args.output or args.input.rsplit('.', 1)[0] + '.json'
            dwg_converter.convert_to_json(
                args.input, output_path,
                include_entities=include_entities,
                include_text_content=True,
                detect_issues=detect_issues
            )
            return

        # JSONL Conversion Mode
        if file_type == 'jsonl':
            logger.info(f"Converting JSONL to text: {args.input}")
            converter = DrawingTextConverter(args.input)
            converter.convert(args.output_text)
            logger.info(f"Conversion complete! Output: {args.output_text}")
            return

        # PDF Processing (original functionality)
        if file_type != 'pdf':
            logger.error(f"Unsupported file type: {args.input}")
            logger.error("Supported formats: PDF, IFC, DWG, DXF, JSONL")
            sys.exit(1)

        # Config Setup
        env_vars = load_env_file(args.env)

        # Async Execution Wrapper
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
                    args.input, client, model, provider,
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
                        logger.info(f"✅ Toon format export complete: {toon_output}")
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
