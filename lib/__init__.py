from pathlib import Path
import logging
import sys

PROJ_ROOT = Path(__file__).parents[1]
DATA_DIR = PROJ_ROOT / "data"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging:
    return logging.getLogger(name)
