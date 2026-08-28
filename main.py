import logging
from datetime import datetime, timezone
from fastapi import FastAPI

# =============================================================
# LOGGING CONFIGURATION
# =============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "logs/ingestion.log",
            encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

# =============================================================
# FASTAPI APPLICATION
# =============================================================

app = FastAPI(
    title="AI RAG Assist",
    description="Retrieval-Augmented Generation backend API",
    version="1.0.0",
)

# =============================================================
# HEALTH API
# =============================================================

@app.get("/health", tags=["Health"], summary="Health check")
def health_check():
    response = {
        "status": "healthy",
        "service": "ai-rag-assist",
        "version": app.version,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }
    logger.info(
        "Health check response: %s",
        response
    )
    return response