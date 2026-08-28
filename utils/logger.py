import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "ingestion.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("ai_rag_assist")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Prevent messages from being passed to the root logger.
logger.propagate = False

# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Console Handler
# ---------------------------------------------------------------------------

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ---------------------------------------------------------------------------
# File Handler
# ---------------------------------------------------------------------------

if LOG_TO_FILE and not any(
    isinstance(handler, logging.FileHandler)
    for handler in logger.handlers
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)