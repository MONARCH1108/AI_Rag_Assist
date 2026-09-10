from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI

from utils.logger import logger
from utils.supabase_service import check_supabase_health
from utils.groq_service import check_groq_health


# =============================================================
# ENVIRONMENT CONFIGURATION
# =============================================================

load_dotenv()


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

@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
)
def health_check():
    logger.info(
        "Starting health check"
    )

    # ---------------------------------------------------------
    # 1. Check Supabase service
    # ---------------------------------------------------------

    supabase_health = check_supabase_health()

    # ---------------------------------------------------------
    # 2. Check Groq service
    # ---------------------------------------------------------

    groq_health = check_groq_health()

    # ---------------------------------------------------------
    # 3. Determine overall application health
    # ---------------------------------------------------------

    if (
        supabase_health["healthy"]
        and groq_health["healthy"]
    ):
        status = "healthy"
    else:
        status = "unhealthy"

    # ---------------------------------------------------------
    # 4. Build health response
    # ---------------------------------------------------------

    response = {
        "status": status,
        "service": "ai-rag-assist",
        "version": app.version,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "dependencies": {
            "supabase": supabase_health,
            "groq": groq_health,
        },
    }

    # ---------------------------------------------------------
    # 5. Log health check result
    # ---------------------------------------------------------

    if status == "healthy":
        logger.info(
            "Health check completed successfully: %s",
            response,
        )
    else:
        logger.error(
            "Health check failed: %s",
            response,
        )

    return response