import argparse
import sys
import os
import json
import asyncio
from datetime import datetime
from typing import Optional
from tqdm import tqdm

from .clients.factory import ClientFactory
from .processing.pdf import PDFProcessor
from .converters.jsonl_to_text import DrawingTextConverter
from .utils.config import load_env_file
from .utils.files import detect_and_download
from .utils.logging import logger

async def process_pdf_async(pdf_path: str, client, model: str, provider: str):
    logger.info(f"Processing: {pdf_path}")
    local_path = detect_and_download(pdf_path)
    
    with PDFProcessor(local_path) as processor:
        page_count = processor.get_page_count()
        logger.info(f"Total pages: {page_count}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_ocr_{timestamp}.json"
        
        logger.info(f"Starting OCR/Analysis with {model}...")
        
        # We process pages sequentially to respect rate limits and memory, 
        # but the architecture supports parallelizing this in future (with semaphores)
        for page_num in tqdm(range(page_count), desc="Processing Pages", unit="page"):
            try:
                # 1. Convert PDF Page to Image
                # Run CPU-bound task in executor if needed, but for now direct call is fine
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
    parser = argparse.ArgumentParser(description="AI Drawing Analyzer 2.0")
    parser.add_argument('pdf', help='PDF file, URL, or JSONL file')
    parser.add_argument('--provider', '-p', choices=list(ClientFactory.PROVIDERS.keys()))
    parser.add_argument('--model', '-m')
    parser.add_argument('--api-key', '-k')
    parser.add_argument('--env', '-e', default='.env')
    parser.add_argument('--to-text', action='store_true', help='Convert JSONL to Text')
    parser.add_argument('--output-text', '-o', default='drawing_complete.txt')
    
    args = parser.parse_args()
    
    # JSONL Conversion Mode
    if args.pdf.endswith('.json') or args.pdf.endswith('.jsonl'):
        converter = DrawingTextConverter(args.pdf)
        converter.convert(args.output_text)
        return

    # Config Setup
    env_vars = load_env_file(args.env)
    
    # Async Execution Wrapper
    async def run_app():
        provider = args.provider
        if not provider:
            provider = await interactive_provider_selection()
            
        # Initialize Client
        try:
            # For interactive model selection, we might need a temporary client or just the provider class logic
            # If model is not provided, we need to instantiate the client to fetch models (for APIs)
            # But for Factory, we usually need API key first.
            
            # Get API Key if not provided
            api_key = args.api_key
            
            # Instantiate client
            # For local, we delay model loading until we know the model, OR we init with default?
            # The Factory logic for local allows model_id=None initially?
            # Actually, my local client loads model in __init__. 
            # So for local, we must know the model BEFORE init.
            
            model = args.model
            client = None
            
            if provider == 'huggingface-local':
                if not model:
                    # We can't init client yet. We simulate model selection.
                    model = await interactive_model_selection(None, provider)
                if not model: return
                client = ClientFactory.create_client(provider, model_id=model)
            else:
                # API Clients
                client = ClientFactory.create_client(provider, api_key)
                if not model:
                    model = await interactive_model_selection(client, provider)
                if not model: return

            output_file = await process_pdf_async(args.pdf, client, model, provider)
            
            if args.to_text and output_file:
                converter = DrawingTextConverter(output_file)
                converter.convert(args.output_text)
                
        except Exception as e:
            logger.error(f"Application Error: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_app())

if __name__ == "__main__":
    main()
