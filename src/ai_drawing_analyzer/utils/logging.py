import logging
import sys
from colorama import Fore, Style, init

init(autoreset=True)

def setup_logger(name: str = "ai_drawing_analyzer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} | "
            f"{Fore.GREEN}%(levelname)s{Style.RESET_ALL} | "
            f"%(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger()
