import os
from typing import Dict

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
        print(f"⚠️ Warning: Could not read {env_path}: {e}")
    
    return env_vars
