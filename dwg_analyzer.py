#!/usr/bin/env python3
"""
Complete PDF Construction Drawing Analyzer CLI
Standalone version with all dependencies integrated
"""

import os
import sys
import json
import base64
import argparse
import httpx
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import io
import re

# PDF and Image Processing
try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# Optional: Transformers and Torch for local Florence-2 support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file"""
    env_vars = {}
    if not os.path.exists(env_path):
        return env_vars
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        env_vars[key] = value
    except Exception as e:
        print(f"⚠️ Warning: Could not read {env_path}: {e}")
    
    return env_vars

def detect_and_download(url_or_path: str, output_dir: str = "tmp_downloads") -> str:
    """Detect if input is a URL and download it."""
    if os.path.exists(url_or_path):
        return url_or_path
    
    if not (url_or_path.startswith('http://') or url_or_path.startswith('https://')):
        return url_or_path
    
    print(f"📥 Detected URL, downloading...")
    os.makedirs(output_dir, exist_ok=True)
    return download_from_url(url_or_path, output_dir)

def download_from_url(url: str, output_dir: str) -> str:
    """Download file from direct URL"""
    try:
        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        
        filename = os.path.basename(url.split('?')[0]) or 'downloaded_file.pdf'
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {filename}")
        return output_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

# ============================================================================
# PDF PROCESSING
# ============================================================================

class PDFProcessor:
    """Process PDF files and detect text vs image pages"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.doc.close()
    
    def get_page_count(self) -> int:
        return len(self.doc)
    
    def get_page_as_image_base64(self, page_num: int) -> str:
        """Get page as base64 encoded image (JPEG for better compression)"""
        page = self.doc[page_num]
        
        # Matrix=2 means 2x zoom (higher resolution) for better OCR accuracy
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=90)
        img_data = img_byte_arr.getvalue()
        return base64.b64encode(img_data).decode('utf-8')

# ============================================================================
# API CLIENTS
# ============================================================================

class HuggingFaceLocalClient:
    """
    Generic Hugging Face Local Client - Works with ANY vision model from Hugging Face Hub
    Supports Florence-2, Qwen-VL, LLaVA, and other vision-language models
    """

    def __init__(self, model_id: str = None):
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise Exception(
                "⚠️ Transformers and Torch required for local inference.\n"
                "Install with: pip install transformers torch\n"
                "For GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )

        self.model_id = model_id or "microsoft/Florence-2-large"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model and processor from Hugging Face Hub"""
        try:
            print(f"📥 Loading model: {self.model_id}")
            print(f"   Device: {self.device.upper()}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

            # Load model with appropriate dtype
            if self.device == "cuda":
                # Use lower precision for GPU to save memory
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # CPU: use default precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                ).to(self.device)

            print(f"✅ Model loaded successfully")

        except ValueError as e:
            error_msg = str(e)
            # Check if it's an architecture mismatch error
            if "does not recognize this architecture" in error_msg or "model type" in error_msg:
                print(f"\n❌ Model architecture not supported by current transformers version")
                print(f"\n💡 SOLUTION: Update transformers to the latest version:")
                print(f"   pip install --upgrade transformers")
                print(f"\nℹ️  Model: {self.model_id}")
                print(f"   Try again after updating transformers")
                raise Exception(f"Unsupported model architecture. Please update transformers: pip install --upgrade transformers")
            else:
                raise Exception(f"Failed to load model '{self.model_id}': {error_msg}")
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                print(f"\n❌ GPU out of memory")
                print(f"\n💡 SOLUTIONS:")
                print(f"   1. Use a smaller model: -m microsoft/Florence-2-base")
                print(f"   2. Use CPU-only mode: CUDA_VISIBLE_DEVICES=\"\" python ...")
                print(f"   3. Reduce image resolution in code")
            raise Exception(f"Failed to load model '{self.model_id}': {error_msg}")

    def get_available_models(self) -> List[Dict[str, str]]:
        """Return popular vision-language models from Hugging Face"""
        return [
            {"id": "microsoft/Florence-2-base", "name": "Florence-2 Base"},
            {"id": "microsoft/Florence-2-large", "name": "Florence-2 Large"},
            {"id": "Qwen/Qwen2-VL-7B-Instruct", "name": "Qwen2-VL 7B"},
            {"id": "Qwen/Qwen3-VL-235B-A22B-Thinking", "name": "Qwen3-VL 235B (Advanced)"},
            {"id": "llava-hf/llava-1.5-7b-hf", "name": "LLaVA 1.5 7B"},
            {"id": "Salesforce/blip2-opt-2.7b", "name": "BLIP-2 OPT 2.7B"},
        ]

    def analyze_image(self, image_base64: str, prompt: str, model: str = None) -> str:
        """Analyze image using local model"""
        try:
            # Convert base64 to PIL Image
            import io
            img_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(img_data))

            # Detect model type and use appropriate task token
            model_id = self.model_id.lower()

            if 'florence' in model_id:
                # Florence-2 specific: Use OCR task token
                task = "<OCR>"
                inputs = self.processor(
                    text=task,
                    images=image,
                    return_tensors="pt"
                ).to(self.device, dtype=self.model.dtype)

                # Generate response
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3,
                        do_sample=False
                    )

                # Decode output with special tokens to get structured format
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False
                )[0]

                # Post-process Florence-2 output to extract structured data
                parsed_answer = self.processor.post_process_generation(
                    generated_text,
                    task=task,
                    image_size=(image.width, image.height)
                )

                # Extract text from parsed answer
                if isinstance(parsed_answer, dict):
                    # For OCR task, the answer contains the extracted text
                    extracted_text = parsed_answer.get('<OCR>', '')
                    return extracted_text if extracted_text else "No text detected in image"
                else:
                    return str(parsed_answer) if parsed_answer else "No text detected in image"
            else:
                # Other models (Qwen-VL, LLaVA, etc.): Use regular prompt
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                # Generate response
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs.get("pixel_values"),
                        max_new_tokens=1024,
                        num_beams=3,
                    )

                # Decode output
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                return generated_text if generated_text else "No response from model"

        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")


class HuggingFaceRouterClient:
    """
    HuggingFace Router Client (OpenAI-compatible)
    Optimized for Qwen2.5-VL (Explicitly supported in your error log)
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.client = httpx.Client(timeout=120.0)
    
    def get_available_models(self) -> List[Dict[str, str]]:
        # Updated based on the error log you provided
        return [
            {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "name": "Qwen2.5-VL 7B (Supported)"},
            {"id": "mistralai/Pixtral-12B-2409", "name": "Pixtral 12B (Supported Backup)"},
            {"id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "name": "NVIDIA Nemotron (Supported Backup)"},
        ]
    
    def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Qwen-VL performs best with specific prompt structuring
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096,  # Increased for full OCR text extraction
            "stream": False
        }
        
        try:
            response = self.client.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 400:
                err_json = response.json()
                msg = err_json.get('error', {}).get('message', str(err_json))
                raise Exception(f"Model Not Supported: {msg}")
            
            # Catch 403/401 errors specifically
            if response.status_code in [401, 403]:
                raise Exception(f"Auth Error ({response.status_code}). Check your HF_TOKEN.")
                
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except httpx.HTTPStatusError as e:
            print(f"\n❌ HF Router Error: {e.response.text}")
            raise Exception(f"HF Router HTTP {e.response.status_code}")


class GeminiClient:
    """Google Gemini API client with Auto-Retry"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.client = httpx.Client(timeout=120.0)
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash (Experimental)"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
        ]
    
    def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }
        
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = self.client.post(url, json=payload)
                
                if response.status_code == 429:
                    print(f"\n⏳ Quota Exceeded (429). Waiting 65 seconds before retry {retry_count + 1}/{max_retries}...")
                    time.sleep(65)
                    retry_count += 1
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if 'promptFeedback' in result and result['promptFeedback'].get('blockReason'):
                    raise Exception(f"Blocked by safety settings: {result['promptFeedback']}")
                
                return result['candidates'][0]['content']['parts'][0]['text']
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    print(f"\n⏳ Quota Exceeded (429). Waiting 65 seconds before retry {retry_count + 1}/{max_retries}...")
                    time.sleep(65)
                    retry_count += 1
                    continue
                print(f"\n❌ Gemini API Error Details: {e.response.text}")
                raise Exception(f"Gemini HTTP {e.response.status_code}")
        
        raise Exception("Max retries exceeded for Gemini API")


class OpenRouterClient:
    """OpenRouter API client"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.Client(timeout=120.0)
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
            {"id": "openai/gpt-4o", "name": "GPT-4o"},
            {"id": "google/gemini-pro-1.5", "name": "Gemini 1.5 Pro"},
        ]
    
    def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            "max_tokens": 2048
        }
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']


class OpenAIClient:
    """OpenAI API client"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.client = httpx.Client(timeout=120.0)
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return [{"id": "gpt-4o", "name": "GPT-4o"}, {"id": "gpt-4o-mini", "name": "GPT-4o Mini"}]
    
    def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            "max_tokens": 2048
        }
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']


class AnthropicClient:
    """Anthropic Claude API client"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.client = httpx.Client(timeout=120.0)
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return [{"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"}, {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"}]
    
    def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        payload = {
            "model": model, "max_tokens": 2048,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                         {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}}]}]
        }
        response = self.client.post(f"{self.base_url}/messages", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['content'][0]['text']

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class PDFAnalyzerCLI:
    """Main CLI application"""
    
    PROVIDERS = {
        'huggingface-local': HuggingFaceLocalClient,
        'huggingface': HuggingFaceRouterClient,
        'openrouter': OpenRouterClient,
        'gemini': GeminiClient,
        'openai': OpenAIClient,
        'anthropic': AnthropicClient
    }

    API_KEY_MAP = {
        'huggingface-local': None,  # No API key needed for local inference
        'huggingface': 'HF_TOKEN',
        'openrouter': 'OPENROUTER_API_KEY',
        'gemini': 'GOOGLE_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY'
    }

    # Models that MUST be run locally (not available via APIs)
    LOCAL_ONLY_MODELS = {
        'microsoft/florence': 'Florence-2 requires local inference',
        'florence-2': 'Florence-2 requires local inference',
        'qwen/qwen2-vl': 'Qwen2-VL requires local inference',
        'qwen2-vl': 'Qwen2-VL requires local inference',
        'llava': 'LLaVA requires local inference',
        'blip2': 'BLIP-2 requires local inference',
        'blip-2': 'BLIP-2 requires local inference',
    }

    def _is_local_only_model(self, model_id: str) -> bool:
        """Check if a model requires local inference"""
        model_lower = model_id.lower()
        for local_pattern in self.LOCAL_ONLY_MODELS.keys():
            if local_pattern in model_lower:
                return True
        return False
    
    def __init__(self, args):
        self.args = args
        self.env_vars = {}
        if args.env:
            self.env_vars = load_env_file(args.env)
            for key, value in self.env_vars.items():
                if key not in os.environ:
                    os.environ[key] = value
    
    def get_api_key(self, provider: str) -> Optional[str]:
        if self.args.api_key: return self.args.api_key
        if provider == 'huggingface-local':
            return None  # No API key needed for local models
        if provider == 'huggingface':
            return os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_API_KEY')
        key_name = self.API_KEY_MAP.get(provider)
        if key_name:
            return os.environ.get(key_name)
        return None
    
    def check_providers(self) -> Dict[str, bool]:
        status = {}
        for provider in self.PROVIDERS.keys():
            status[provider] = bool(self.get_api_key(provider))
        return status
    
    def select_provider_interactive(self) -> str:
        status = self.check_providers()
        print("\n🤖 Available AI Providers:")
        print("=" * 70)
        providers = list(self.PROVIDERS.keys())

        provider_descriptions = {
            'huggingface-local': '(Local - Any HF model, requires transformers & torch)',
            'huggingface': '(API - Free tier with Qwen2.5-VL, etc.)',
            'openrouter': '(API - Multi-model access)',
            'gemini': '(API - Google Gemini)',
            'openai': '(API - OpenAI GPT-4o)',
            'anthropic': '(API - Claude Sonnet/Haiku)'
        }

        for i, provider in enumerate(providers, 1):
            indicator = "✅" if status[provider] else "❌"
            desc = provider_descriptions.get(provider, '')
            print(f"{i}. {indicator} {provider.title():25} {desc}")
        print("=" * 70)

        while True:
            try:
                choice = input(f"\nSelect provider (1-{len(providers)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(providers):
                    provider = providers[idx]
                    if not status[provider] and provider != 'huggingface-local':
                        print(f"❌ {provider.title()} is not configured. Please check .env file.")
                        continue
                    return provider
                print(f"Invalid choice. Please enter 1-{len(providers)}.")
            except (ValueError, KeyboardInterrupt):
                sys.exit(0)

    def _extract_hf_model_id(self, user_input: str) -> str:
        user_input = user_input.strip()
        if "huggingface.co/" in user_input:
            parts = user_input.split("huggingface.co/")
            return parts[1].strip("/")
        return user_input
    
    def select_model_interactive(self, client, provider: str) -> str:
        """Interactive model selection with Custom Option for HF (both router and local)"""

        # For local provider, offer quick model input
        if provider == 'huggingface-local':
            if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
                print("\n❌ Missing required packages for local inference!")
                print("Install with: pip install transformers torch")
                return None

            print(f"\n🤖 Local Model Inference (Hugging Face)")
            print("=" * 70)
            print("\n📋 Popular models:")
            models = [
                {"id": "microsoft/Florence-2-large", "name": "Florence-2 Large (best OCR, ~770M)"},
                {"id": "microsoft/Florence-2-base", "name": "Florence-2 Base (lighter, ~300M)"},
                {"id": "Qwen/Qwen2-VL-7B-Instruct", "name": "Qwen2-VL 7B (strong vision)"},
                {"id": "llava-hf/llava-1.5-7b-hf", "name": "LLaVA 1.5 7B (general purpose)"},
                {"id": "Salesforce/blip2-opt-2.7b", "name": "BLIP-2 OPT 2.7B (lightweight)"},
            ]
            for i, model in enumerate(models, 1):
                print(f"{i}. {model['name']}")
                print(f"   ID: {model['id']}")

            print("-" * 70)
            print("0. [CUSTOM] Enter any Hugging Face model ID or URL")
            print("=" * 70)

            while True:
                try:
                    choice = input(f"\nSelect model (1-5) or '0' for custom or 'q' to quit: ").strip()

                    if choice.lower() == 'q':
                        sys.exit(0)

                    if choice == '0':
                        custom_input = input("\n🔗 Paste Model ID or URL (e.g., microsoft/Florence-2-large): ").strip()
                        if not custom_input:
                            print("❌ Empty input. Try again.")
                            continue
                        model_id = self._extract_hf_model_id(custom_input)
                        return model_id

                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        return models[idx]['id']
                    else:
                        print(f"Invalid choice. Please enter 1-5 or 0.")
                except (ValueError, KeyboardInterrupt):
                    sys.exit(0)

        print(f"\n🔍 Fetching available models for {provider.title()}...")

        try:
            # Get available models - create temporary client instance if needed
            if provider == 'huggingface':
                if client is None:
                    api_key = self.get_api_key(provider)
                    client = self.PROVIDERS[provider](api_key)
                models = client.get_available_models()
            else:
                if client is None:
                    api_key = self.get_api_key(provider)
                    client = self.PROVIDERS[provider](api_key)
                models = client.get_available_models()

            print(f"\n📋 Available Models:")
            print("=" * 70)

            for i, model in enumerate(models, 1):
                is_local_only = self._is_local_only_model(model['id']) if provider != 'huggingface-local' else False
                local_note = " 🖥️  (Local only)" if is_local_only else ""
                print(f"{i}. {model['name']}{local_note}")
                print(f"   ID: {model['id']}")

            if provider in ['huggingface', 'huggingface-local']:
                print("-" * 70)
                print("0. [CUSTOM] Enter custom Hugging Face Model ID or URL")

            print("=" * 70)

            while True:
                try:
                    choice = input(f"\nSelect model (1-{len(models)})" +
                                   (" or '0' for custom" if provider in ['huggingface', 'huggingface-local'] else "") +
                                   " or 'q' to quit: ").strip()

                    if choice.lower() == 'q':
                        sys.exit(0)

                    if provider in ['huggingface', 'huggingface-local'] and choice == '0':
                        custom_input = input("\n🔗 Paste Model ID or URL (e.g., microsoft/Florence-2-large or https://huggingface.co/...): ").strip()
                        if not custom_input:
                            print("❌ Empty input. Try again.")
                            continue
                        model_id = self._extract_hf_model_id(custom_input)
                        return model_id

                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        return models[idx]['id']
                    else:
                        print(f"Invalid choice.")
                except (ValueError, KeyboardInterrupt):
                    sys.exit(0)

        except Exception as e:
            print(f"❌ Error fetching models: {e}")
            return None
    
    def process_pdf(self, pdf_path: str, client, model: str, provider: str):
        print(f"\n📄 Processing: {pdf_path}")
        local_path = detect_and_download(pdf_path)
        
        with PDFProcessor(local_path) as processor:
            page_count = processor.get_page_count()
            print(f"📊 Total pages: {page_count}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output_ocr_{timestamp}.json"
            
            print(f"\n🚀 Starting OCR/Analysis with {model}...")
            print("=" * 70)
            
            for page_num in range(page_count):
                print(f"\n📄 Page {page_num + 1}/{page_count}")
                try:
                    # 1. Convert PDF Page to Image (PyMuPDF)
                    image_base64 = processor.get_page_as_image_base64(page_num)
                    
                    # 2. Define OCR Prompt for the Model
                    prompt = "OCR Task: Transcribe all text visible in this image verbatim. Do not summarize or analyze. Output only the text."
                    
                    print("🔍 Extracting text (OCR)...")
                    
                    # 3. Model analyzes image and outputs text
                    response = client.analyze_image(image_base64, prompt, model)
                    
                    preview = response[:150].replace('\n', ' ') + "..." if len(response) > 150 else response
                    print(f"✅ Extracted: {preview}")
                    
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
                    print(f"❌ Error on page {page_num + 1}: {e}")
                    if 'Model Not Supported' in str(e):
                        print("💡 TIP: The selected model is not supported by the provider.")
                        print("   Supported models from your log: Qwen/Qwen2.5-VL-7B-Instruct, mistralai/Pixtral-12B-2409")
                        return
                    if '410' in str(e) or '404' in str(e): 
                        print("CRITICAL: API Endpoint error. Model might be offline.")
                        return
                    
            print(f"\n✅ OCR complete! Saved to {output_file}")
    
    def run(self):
        print("🚀 PDF OCR & Analyzer")
        print("=" * 70)

        if self.args.provider:
            provider = self.args.provider.lower()
        else:
            provider = self.select_provider_interactive()

        if self.args.model:
            model = self.args.model
        else:
            model = self.select_model_interactive(None, provider)
            if not model: sys.exit(1)

        # Validate: Check if user is trying to use a local-only model with an API provider
        if self._is_local_only_model(model) and provider != 'huggingface-local':
            print("\n⚠️  WARNING: This model requires local inference!")
            print(f"📍 Model: {model}")
            print(f"❌ Cannot use with provider: {provider}")
            print("\n💡 SOLUTION: Use the 'huggingface-local' provider instead")
            print(f"   Command: python pdf_analyzer_complete.py {self.args.pdf} -p huggingface-local -m {model}")
            print("\n📦 Installation: pip install transformers torch")
            print("🚀 For GPU (faster): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            sys.exit(1)

        # Initialize client with model_id for local inference
        api_key = self.get_api_key(provider)
        if provider == 'huggingface-local':
            client = self.PROVIDERS[provider](model)
        else:
            client = self.PROVIDERS[provider](api_key)

        self.process_pdf(self.args.pdf, client, model, provider)

        # Optionally convert to text for LLM
        if self.args.to_text:
            self._convert_to_text()

    def _convert_to_text(self):
        """Convert the most recent JSONL output to text format"""
        import glob

        # Find the most recent output_ocr file
        output_files = glob.glob('output_ocr_*.json')
        if not output_files:
            print("\n⚠️  No OCR output file found. Skipping text conversion.")
            return

        latest_output = sorted(output_files)[-1]
        print(f"\n✅ Found OCR output: {latest_output}")

        try:
            converter = DrawingTextConverter(latest_output)
            text_file = converter.convert(self.args.output_text)
            print(f"\n✨ Text conversion complete: {text_file}")
            print(f"   Ready to feed into Claude or other LLMs!")
        except Exception as e:
            print(f"\n❌ Text conversion failed: {e}")

# ============================================================================
# DRAWING TEXT CONVERTER FOR LLM CONSUMPTION
# ============================================================================

class DrawingTextConverter:
    """Convert JSONL OCR output to LLM-friendly text format"""

    def __init__(self, jsonl_file: str):
        self.jsonl_file = jsonl_file
        self.pages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, str] = {}
        self.cross_refs: Dict[str, List[int]] = {}
        self.legends: Dict[str, str] = {}
        self.document_structure: str = ""

        # Regex patterns for extraction
        self.patterns = {
            'page_reference': [
                r'(?:see|refer to|detail|section|drawing)\s+([A-Z0-9\-]+).*?(?:page|pg\.?)\s*(\d+)',
                r'(?:page|pg\.?)\s*(\d+)\s+.*?(?:detail|section|schedule)\s+([A-Z0-9\-]+)',
                r'(?:see|ref(?:er)?\s+to)\s+(?:page|pg\.?)\s*(\d+)',
                r'on\s+(?:page|pg\.?)\s*(\d+)',
                r'(?:page|pg\.?)\s*(\d+)',
            ],
            'title_block': {
                'project': r'(?:project|proj\.?|project\s+name)[\s:]*([^\n]+)',
                'title': r'(?:title|drawing\s+title|drawing\s+name)[\s:]*([^\n]+)',
                'discipline': r'(?:discipline)[\s:]*([^\n]+)',
                'sheet': r'(?:sheet|drawing\s+number|sheet\s+number)[\s:]*([A-Z0-9\-\.]+)',
                'revision': r'(?:revision|rev\.?)[\s:]*([A-Z0-9\-]+)',
                'date': r'(?:date|rev\.?\s+date|revision\s+date)[\s:]*([0-9\-/]+)',
                'scale': r'(?:scale)[\s:]*([0-9\':\" =/\-]+)',
                'drawn_by': r'(?:drawn\s+by|drawn)[\s:]*([^\n]+)',
                'checked_by': r'(?:checked\s+by|checked)[\s:]*([^\n]+)',
                'approved_by': r'(?:approved\s+by|approved)[\s:]*([^\n]+)',
            },
            'legend_keywords': ['legend', 'symbols', 'abbreviations', 'notes', 'key', 'symbol key']
        }

    def load_pages(self) -> List[Dict[str, Any]]:
        """Load all pages from JSONL file"""
        try:
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            page_data = json.loads(line)
                            self.pages.append(page_data)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Warning: Skipped malformed JSON entry: {e}")

            print(f"✅ Loaded {len(self.pages)} pages from {self.jsonl_file}")
            return self.pages
        except FileNotFoundError:
            print(f"❌ File not found: {self.jsonl_file}")
            raise

    def extract_metadata(self) -> Dict[str, str]:
        """Extract title block info from first page"""
        if not self.pages:
            print("⚠️ No pages loaded. Returning empty metadata.")
            return {}

        first_page_text = self.pages[0].get('text_content', '')
        self.metadata = {
            'project': 'UNKNOWN',
            'title': 'UNKNOWN',
            'discipline': 'UNKNOWN',
            'sheet': 'UNKNOWN',
            'revision': 'UNKNOWN',
            'date': 'UNKNOWN',
            'scale': 'UNKNOWN',
            'drawn_by': 'UNKNOWN',
            'checked_by': 'UNKNOWN',
            'approved_by': 'UNKNOWN',
        }

        # Try to extract metadata using regex
        for field, pattern in self.patterns['title_block'].items():
            match = re.search(pattern, first_page_text, re.IGNORECASE | re.MULTILINE)
            if match:
                self.metadata[field] = match.group(1).strip()

        return self.metadata

    def find_cross_references(self) -> Dict[str, List[int]]:
        """Build cross-reference map from all pages"""
        self.cross_refs = {}

        for page_idx, page in enumerate(self.pages):
            text = page.get('text_content', '')
            page_num = page_idx + 1

            # Find all page references in this page
            for pattern in self.patterns['page_reference']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if groups:
                        # Extract the referenced detail/section name and page number
                        if len(groups) >= 2:
                            detail_name = groups[0].strip() if groups[0] else f"Reference on page {page_num}"
                            page_ref = int(groups[1]) if groups[1].isdigit() else page_num
                        else:
                            page_ref = int(groups[0]) if groups[0] and groups[0].isdigit() else page_num
                            detail_name = f"Reference on page {page_num}"

                        if detail_name not in self.cross_refs:
                            self.cross_refs[detail_name] = []
                        if page_ref not in self.cross_refs[detail_name]:
                            self.cross_refs[detail_name].append(page_ref)

        # Sort page numbers in each reference
        for key in self.cross_refs:
            self.cross_refs[key].sort()

        return self.cross_refs

    def extract_legends(self) -> Dict[str, str]:
        """Find and extract legend/symbol definitions"""
        self.legends = {}

        for page_idx, page in enumerate(self.pages):
            text = page.get('text_content', '')
            page_num = page_idx + 1

            # Check if this page contains legend keywords
            is_legend_page = any(keyword.lower() in text.lower()
                                for keyword in self.patterns['legend_keywords'])

            if is_legend_page:
                # Extract legend content - store the entire page as legend reference
                legend_key = f"Page {page_num} - Symbols/Legend"
                self.legends[legend_key] = text

        return self.legends

    def create_document_structure(self) -> str:
        """Create table of contents / structure map"""
        if not self.pages:
            return "No pages available"

        structure = "DOCUMENT STRUCTURE:\n"
        structure += "=" * 80 + "\n"

        # Summarize first few lines of each page
        for page_idx, page in enumerate(self.pages):
            page_num = page_idx + 1
            text = page.get('text_content', '').strip()

            # Get first meaningful line(s) as summary
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            summary = lines[0][:60] if lines else "Content not available"

            structure += f"Page {page_num}: {summary}\n"

        structure += "=" * 80 + "\n"
        self.document_structure = structure
        return structure

    def assemble_output(self) -> str:
        """Combine all sections into final text"""
        output_parts = []

        # Header
        output_parts.append("=" * 80)
        output_parts.append("COMPLETE DRAWING SET - TEXT EXTRACTION FOR LLM ANALYSIS")
        output_parts.append("=" * 80)
        output_parts.append("")

        # Metadata Section
        output_parts.append("DOCUMENT METADATA")
        output_parts.append("=" * 80)
        for key, value in self.metadata.items():
            display_key = key.replace('_', ' ').title()
            output_parts.append(f"{display_key}: {value}")
        output_parts.append(f"Total Pages: {len(self.pages)}")
        output_parts.append(f"Extracted Date: {datetime.now().isoformat()}Z")
        output_parts.append("=" * 80)
        output_parts.append("")

        # Cross-Reference Index
        if self.cross_refs:
            output_parts.append("CROSS-REFERENCE INDEX")
            output_parts.append("=" * 80)
            for detail, pages in sorted(self.cross_refs.items()):
                pages_str = ", ".join(str(p) for p in sorted(set(pages)))
                output_parts.append(f"{detail}: pages {pages_str}")
            output_parts.append("=" * 80)
            output_parts.append("")

        # Legend Section
        if self.legends:
            output_parts.append("LEGEND & SYMBOLS REFERENCE")
            output_parts.append("=" * 80)
            for legend_name, legend_content in self.legends.items():
                output_parts.append(f"\n{legend_name}:")
                output_parts.append("-" * 80)
                output_parts.append(legend_content[:500])  # Limit to first 500 chars
                output_parts.append("")
            output_parts.append("=" * 80)
            output_parts.append("")

        # Document Structure
        output_parts.append(self.document_structure)
        output_parts.append("")

        # Complete Drawing Content
        output_parts.append("COMPLETE DRAWING CONTENT")
        output_parts.append("=" * 80)
        output_parts.append("")

        for page_idx, page in enumerate(self.pages):
            page_num = page_idx + 1
            total_pages = len(self.pages)

            output_parts.append("-" * 80)
            output_parts.append(f"PAGE {page_num} of {total_pages}")
            output_parts.append("-" * 80)
            output_parts.append(page.get('text_content', ''))
            output_parts.append("")
            output_parts.append("")

        # Footer
        output_parts.append("=" * 80)
        output_parts.append("END OF DRAWING SET")
        output_parts.append(f"Generated: {datetime.now().isoformat()}Z")
        output_parts.append(f"Total Characters: {sum(len(p) for p in output_parts)}")
        output_parts.append(f"Total Pages: {len(self.pages)}")
        output_parts.append(f"Cross-References Found: {len(self.cross_refs)}")
        output_parts.append("=" * 80)

        return "\n".join(output_parts)

    def save_to_file(self, output_path: str) -> None:
        """Write complete text to file"""
        try:
            content = self.assemble_output()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving to file: {e}")
            raise

    def convert(self, output_path: str = 'drawing_complete.txt') -> str:
        """Main conversion pipeline"""
        try:
            print(f"\n📄 Converting JSONL to text format...")
            print("=" * 70)

            # 1. Load pages
            print("📥 Loading pages from JSONL...")
            self.load_pages()

            # 2. Extract metadata
            print("🔍 Extracting metadata...")
            self.extract_metadata()

            # 3. Find cross-references
            print("🔗 Finding cross-references...")
            self.find_cross_references()

            # 4. Extract legends
            print("📋 Extracting legends...")
            self.extract_legends()

            # 5. Create structure
            print("📊 Creating document structure...")
            self.create_document_structure()

            # 6. Assemble and save
            print("🔨 Assembling output...")
            self.save_to_file(output_path)

            # Summary
            print("\n" + "=" * 70)
            print(f"✅ Conversion Complete!")
            print(f"   📊 Pages processed: {len(self.pages)}")
            print(f"   🔗 Cross-references found: {len(self.cross_refs)}")
            print(f"   📋 Legend sections found: {len(self.legends)}")
            print("=" * 70)

            return output_path
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="OCR and Analyze PDFs with AI")
    parser.add_argument('pdf', help='PDF file path, URL, or JSONL file for conversion')
    parser.add_argument('--provider', '-p', choices=['huggingface-local', 'huggingface', 'openrouter', 'gemini', 'openai', 'anthropic'],
                        help='AI provider to use (huggingface-local runs models locally)')
    parser.add_argument('--model', '-m', help='Model ID to use (optional)')
    parser.add_argument('--api-key', '-k', help='API key (not needed for huggingface-local)')
    parser.add_argument('--env', '-e', default='.env', help='Path to .env file')
    parser.add_argument('--to-text', '--convert-to-text', action='store_true',
                        help='Convert JSONL OCR output to complete text file for LLM consumption')
    parser.add_argument('--output-text', '-o', default='drawing_complete.txt',
                        help='Output filename for text conversion (default: drawing_complete.txt)')

    args = parser.parse_args()

    # Handle JSONL conversion request
    if args.pdf.endswith('.json') or args.pdf.endswith('.jsonl'):
        # This is a conversion request
        if not os.path.exists(args.pdf):
            print(f"❌ File not found: {args.pdf}")
            sys.exit(1)

        print(f"🔄 JSONL to Text Conversion Mode")
        converter = DrawingTextConverter(args.pdf)
        try:
            text_file = converter.convert(args.output_text)
            print(f"\n✅ Conversion complete: {text_file}")
            print(f"   Ready to feed into Claude or other LLMs!")
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            sys.exit(1)
        sys.exit(0)

    # Normal PDF processing
    app = PDFAnalyzerCLI(args)
    app.run()

if __name__ == "__main__":
    main()