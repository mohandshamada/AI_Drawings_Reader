import os
import platform
from typing import Dict, Any, Tuple


def get_available_memory() -> Tuple[float, float]:
    """
    Get available system memory in GB.

    Returns:
        Tuple of (total_ram_gb, available_ram_gb)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        return total_gb, available_gb
    except ImportError:
        pass

    # Fallback for systems without psutil
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            total = available = 0
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total = int(line.split()[1]) / (1024 ** 2)  # KB to GB
                elif line.startswith('MemAvailable:'):
                    available = int(line.split()[1]) / (1024 ** 2)
            if total > 0:
                return total, available if available > 0 else total * 0.5
        elif platform.system() == "Darwin":  # macOS
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                total = int(result.stdout.strip()) / (1024 ** 3)
                return total, total * 0.5  # Assume 50% available
    except Exception:
        pass

    # Default fallback: assume 8GB total, 4GB available
    return 8.0, 4.0


def get_gpu_memory() -> float:
    """
    Get available GPU memory in GB (NVIDIA only).

    Returns:
        GPU memory in GB, or 0 if no GPU available
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get free memory on first GPU
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            free = (total - reserved) / (1024 ** 3)
            return free
    except ImportError:
        pass

    # Try nvidia-smi as fallback
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Get first GPU's free memory (in MB)
            free_mb = int(result.stdout.strip().split('\n')[0])
            return free_mb / 1024
    except Exception:
        pass

    return 0.0


def get_optimal_settings() -> Dict[str, int]:
    """
    Determine optimal PDF processing settings based on available resources.

    Returns:
        Dict with 'pdf_zoom_level' and 'jpeg_quality' values
    """
    total_ram, available_ram = get_available_memory()
    gpu_memory = get_gpu_memory()

    # Use the minimum of total RAM and available RAM for decision
    effective_memory = min(total_ram, available_ram * 1.5)

    # If GPU is available with good memory, we can be more aggressive
    if gpu_memory >= 8:
        effective_memory = max(effective_memory, gpu_memory * 2)

    # Determine settings based on effective memory
    if effective_memory < 4:
        # Low memory: conservative settings
        zoom_level = 1
        jpeg_quality = 70
    elif effective_memory < 8:
        # Medium memory: balanced settings
        zoom_level = 2
        jpeg_quality = 85
    elif effective_memory < 16:
        # High memory: quality settings
        zoom_level = 3
        jpeg_quality = 90
    else:
        # Very high memory: maximum quality
        zoom_level = 4
        jpeg_quality = 95

    return {
        'pdf_zoom_level': zoom_level,
        'jpeg_quality': jpeg_quality,
        'detected_ram_gb': round(total_ram, 1),
        'detected_available_gb': round(available_ram, 1),
        'detected_gpu_gb': round(gpu_memory, 1) if gpu_memory > 0 else None
    }


class AppConfig:
    """Application configuration with defaults"""

    # These will be set dynamically based on resources
    _optimal_settings = None

    # PDF Processing (defaults, may be overridden by auto-detection)
    PDF_ZOOM_LEVEL = 2  # 2x zoom for high-resolution conversion
    JPEG_QUALITY = 90  # JPEG quality (1-100)

    # API & Network
    API_TIMEOUT = 120.0  # seconds
    DOWNLOAD_TIMEOUT = 60  # seconds
    MAX_RETRIES = 3
    RETRY_BACKOFF_MULTIPLIER = 1  # exponential backoff: base * multiplier^attempt

    # Processing
    MAX_TOKENS = 2048  # Max tokens in LLM response
    BATCH_SIZE = 1  # Pages processed at once (1 = sequential)

    # Output
    ENABLE_COMPRESSION = False  # Compress output JSON
    CACHE_DOWNLOADS = True  # Cache downloaded PDFs

    @classmethod
    def _get_optimal_settings(cls) -> Dict[str, int]:
        """Get optimal settings (cached)"""
        if cls._optimal_settings is None:
            cls._optimal_settings = get_optimal_settings()
        return cls._optimal_settings

    @classmethod
    def from_env(cls, auto_adjust: bool = True) -> Dict[str, Any]:
        """
        Load config from environment variables.

        Args:
            auto_adjust: If True, auto-detect optimal settings based on resources
                        when not explicitly set in environment
        """
        # Get auto-detected optimal settings
        optimal = cls._get_optimal_settings() if auto_adjust else {}

        # Environment variables override auto-detection
        # If env var is set, use it; otherwise use optimal (if available) or class default
        pdf_zoom = os.getenv('PDF_ZOOM_LEVEL')
        jpeg_quality = os.getenv('JPEG_QUALITY')

        config = {
            'pdf_zoom_level': int(pdf_zoom) if pdf_zoom else optimal.get('pdf_zoom_level', cls.PDF_ZOOM_LEVEL),
            'jpeg_quality': int(jpeg_quality) if jpeg_quality else optimal.get('jpeg_quality', cls.JPEG_QUALITY),
            'api_timeout': float(os.getenv('API_TIMEOUT', cls.API_TIMEOUT)),
            'download_timeout': float(os.getenv('DOWNLOAD_TIMEOUT', cls.DOWNLOAD_TIMEOUT)),
            'max_retries': int(os.getenv('MAX_RETRIES', cls.MAX_RETRIES)),
            'max_tokens': int(os.getenv('MAX_TOKENS', cls.MAX_TOKENS)),
            'batch_size': int(os.getenv('BATCH_SIZE', cls.BATCH_SIZE)),
        }

        # Add resource info for debugging
        if auto_adjust and optimal:
            config['_auto_adjusted'] = True
            config['_detected_ram_gb'] = optimal.get('detected_ram_gb')
            config['_detected_gpu_gb'] = optimal.get('detected_gpu_gb')

        return config

    @classmethod
    def get_resource_info(cls) -> str:
        """Get a human-readable string of detected resources"""
        optimal = cls._get_optimal_settings()
        total_ram = optimal.get('detected_ram_gb', 0)
        available_ram = optimal.get('detected_available_gb', 0)
        gpu_mem = optimal.get('detected_gpu_gb')

        info = f"RAM: {available_ram:.1f}GB available / {total_ram:.1f}GB total"
        if gpu_mem:
            info += f", GPU: {gpu_mem:.1f}GB"
        info += f" -> zoom={optimal['pdf_zoom_level']}, quality={optimal['jpeg_quality']}"
        return info


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
                        # Also set in os.environ if not present
                        if key not in os.environ:
                            os.environ[key] = value
    except Exception as e:
        print(f"Warning: Could not read {env_path}: {e}")

    return env_vars
