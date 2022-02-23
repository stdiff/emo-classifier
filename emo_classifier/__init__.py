import logging
import sys
from pathlib import Path
from typing import Optional

LIB_ROOT = Path(__file__).parent
RESOURCES_DIR = LIB_ROOT / "resources"
ARTIFACT_DIR = LIB_ROOT / "artifact"


def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout) if log_file is None else logging.FileHandler(str(log_file))
    format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(format)
    logger.addHandler(handler)
    return logger
