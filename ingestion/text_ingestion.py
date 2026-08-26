from pathlib import Path
import logging
import os

# =============================================================
# LOGGING CONFIGURATION
# =============================================================

LOG_TO_FILE = os.getenv("LOG_TO_FILE").lower() == "true"
LOG_HANDLERS = [
    logging.StreamHandler(),
]
if LOG_TO_FILE:
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    LOG_FILE = LOG_DIR / "ingestion.log"
    LOG_HANDLERS.append(
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=LOG_HANDLERS,
)
logger = logging.getLogger(__name__)


def detect_file_type(file_path):
    """
    Detect the document type using the file extension.

    Args:
        file_path (str): Path to the document.

    Returns:
        dict: Structured file detection result.
    """
    path = Path(file_path)
    logger.info("Starting file type detection: %s", path)

    # ---------------------------------------------------------
    # 1. Validate that the file exists
    # ---------------------------------------------------------
    if not path.exists():
        logger.error("File not found: %s", path)

        return {
            "success": False,
            "file_name": path.name,
            "file_type": None,
            "file_extension": None,
            "error": {
                "type": "FILE_NOT_FOUND",
                "message": f"The file was not found: {path}"
            }
        }

    # ---------------------------------------------------------
    # 2. Validate that the path points to a file
    # ---------------------------------------------------------
    if not path.is_file():
        logger.error("Path is not a file: %s", path)

        return {
            "success": False,
            "file_name": path.name,
            "file_type": None,
            "file_extension": None,
            "error": {
                "type": "INVALID_FILE",
                "message": f"The provided path is not a file: {path}"
            }
        }

    # ---------------------------------------------------------
    # 3. Get the file extension
    # ---------------------------------------------------------
    file_extension = path.suffix.lower()

    logger.info(
        "File extension detected: %s",
        file_extension or "none"
    )

    # ---------------------------------------------------------
    # 4. Map extension to document type
    # ---------------------------------------------------------
    file_type_map = {
        ".pdf": "pdf",
        ".doc": "doc",
        ".docx": "docx",
        ".xls": "excel",
        ".xlsx": "excel",
        ".csv": "csv",
        ".json": "json",
        ".txt": "text",
        ".md": "markdown",
    }

    file_type = file_type_map.get(file_extension)

    # ---------------------------------------------------------
    # 5. Reject unsupported file types
    # ---------------------------------------------------------
    if file_type is None:
        logger.warning(
            "Unsupported file type: %s",
            file_extension or "unknown"
        )

        return {
            "success": False,
            "file_name": path.name,
            "file_type": "unknown",
            "file_extension": file_extension or None,
            "error": {
                "type": "UNSUPPORTED_FILE_TYPE",
                "message": (
                    f"The file type '{file_extension or 'unknown'}' "
                    "is not currently supported."
                )
            }
        }

    # ---------------------------------------------------------
    # 6. Successful detection
    # ---------------------------------------------------------
    logger.info(
        "File type detected successfully: %s -> %s",
        path.name,
        file_type
    )

    return {
        "success": True,
        "file_name": path.name,
        "file_type": file_type,
        "file_extension": file_extension,
        "error": None
    }


if __name__ == "__main__":
    file_path = input("Enter the path to the document: ").strip()

    response = detect_file_type(file_path)

    print(response)