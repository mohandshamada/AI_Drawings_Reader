from typing import List, Dict, Optional
import httpx
import base64
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from .base import BaseClient
from ..utils.logging import logger

# Check for local deps
try:
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        AutoModelForVision2Seq,
        Qwen2VLForConditionalGeneration,
    )
    LOCAL_DEPS_AVAILABLE = True
except ImportError:
    LOCAL_DEPS_AVAILABLE = False


class HuggingFaceRouterClient(BaseClient):
    """HuggingFace Router Client (OpenAI-compatible)"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
    
    async def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "name": "Qwen2.5-VL 7B (Supported)"},
            {"id": "mistralai/Pixtral-12B-2409", "name": "Pixtral 12B (Supported Backup)"},
            {"id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "name": "NVIDIA Nemotron (Supported Backup)"},
        ]
    
    async def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
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
            "max_tokens": 4096,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.api_url, headers=headers, json=payload)
                
                if response.status_code == 400:
                    err_json = response.json()
                    msg = err_json.get('error', {}).get('message', str(err_json))
                    raise Exception(f"Model Not Supported: {msg}")
                
                if response.status_code in [401, 403]:
                    raise Exception(f"Auth Error ({response.status_code}). Check your HF_TOKEN.")
                    
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except httpx.HTTPStatusError as e:
                raise Exception(f"HF Router HTTP {e.response.status_code}: {e.response.text}")


class HuggingFaceLocalClient(BaseClient):
    """Local Inference Client"""

    def __init__(self, model_id: str = None):
        super().__init__(None) # No API Key
        if not LOCAL_DEPS_AVAILABLE:
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
        try:
            logger.info(f"📥 Loading model: {self.model_id}")
            logger.info(f"   Device: {self.device.upper()}")

            # Load processor (trust_remote_code only for models that need it)
            model_lower = self.model_id.lower()
            if 'qwen' in model_lower or 'florence' in model_lower:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )
            else:
                self.processor = AutoProcessor.from_pretrained(self.model_id)

            # Determine correct model class based on model type
            if 'qwen2-vl' in model_lower or 'qwen2_vl' in model_lower:
                # Qwen2-VL models require specific class
                ModelClass = Qwen2VLForConditionalGeneration
                logger.info("   Using Qwen2VLForConditionalGeneration")
            elif 'florence' in model_lower:
                # Florence models use AutoModelForCausalLM
                ModelClass = AutoModelForCausalLM
                logger.info("   Using AutoModelForCausalLM")
            elif 'llava' in model_lower or 'vision' in model_lower:
                # LLaVA and other vision models
                ModelClass = AutoModelForVision2Seq
                logger.info("   Using AutoModelForVision2Seq")
            else:
                # Default to CausalLM for other models
                ModelClass = AutoModelForCausalLM
                logger.info("   Using AutoModelForCausalLM (default)")

            # Load model with appropriate settings
            if self.device == "cuda":
                try:
                    # Try with device_map (requires accelerate)
                    self.model = ModelClass.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except ImportError as e:
                    logger.warning(f"   Accelerate not available, loading without device_map")
                    logger.warning(f"   Install with: pip install 'accelerate>=0.26.0'")
                    # Fallback: load without device_map
                    self.model = ModelClass.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    ).to(self.device)
            else:
                self.model = ModelClass.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                ).to(self.device)

            logger.info(f"✅ Model loaded successfully")
        except Exception as e:
            error_msg = str(e)
            if "accelerate" in error_msg.lower():
                raise Exception(
                    f"Failed to load model '{self.model_id}': Missing 'accelerate' package.\n"
                    f"Install with: pip install 'accelerate>=0.26.0'\n"
                    f"Or reinstall with: pip install -e '.[local]'"
                )
            raise Exception(f"Failed to load model '{self.model_id}': {error_msg}")

    async def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "microsoft/Florence-2-base", "name": "Florence-2 Base"},
            {"id": "microsoft/Florence-2-large", "name": "Florence-2 Large"},
            {"id": "Qwen/Qwen2-VL-7B-Instruct", "name": "Qwen2-VL 7B"},
            {"id": "llava-hf/llava-1.5-7b-hf", "name": "LLaVA 1.5 7B"},
        ]

    async def analyze_image(self, image_base64: str, prompt: str, model: str = None) -> str:
        """Analyze image using local model inference (runs in thread pool for async compatibility)"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self._run_inference, image_base64, prompt)

    def _run_inference(self, image_base64: str, prompt: str) -> str:
        """Run synchronous model inference"""
        try:
            img_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(img_data))
            model_id = self.model_id.lower()

            if 'florence' in model_id:
                # Florence-2 models use task prompts
                task = "<OCR>"
                inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device, dtype=self.model.dtype)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3,
                        do_sample=False
                    )

                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text,
                    task=task,
                    image_size=(image.width, image.height)
                )

                if isinstance(parsed_answer, dict):
                    extracted_text = parsed_answer.get('<OCR>', '')
                    return extracted_text if extracted_text else "No text detected in image"
                else:
                    return str(parsed_answer)

            elif 'qwen2-vl' in model_id or 'qwen2_vl' in model_id:
                # Qwen2-VL models use chat format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False
                    )

                # Trim input tokens from output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                return output_text[0]

            else:
                # Generic vision-language models
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        num_beams=3,
                    )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text

        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")
