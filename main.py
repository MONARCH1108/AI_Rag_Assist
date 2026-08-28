from datetime import datetime, timezone
from fastapi import FastAPI
from utils.logger import logger
from utils.supabase_service import check_supabase_health

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
    # 2. Determine overall application health
    # ---------------------------------------------------------

    if supabase_health["healthy"]:
        status = "healthy"
    else:
        status = "unhealthy"

    # ---------------------------------------------------------
    # 3. Build health response
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
        },
    }

    # ---------------------------------------------------------
    # 4. Log health check result
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
