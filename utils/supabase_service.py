import os
from supabase import Client, create_client
from utils.logger import logger

# =============================================================
# SUPABASE SERVICE CONFIGURATION
# =============================================================
DOCUMENTS_TABLE = "documents"
_supabase_client = None

def get_supabase_client():
    """
    Get the shared Supabase client.

    The client is created only once and reused for
    subsequent calls.

    Returns:
        Client:
            Configured Supabase client.

    Raises:
        ValueError:
            If Supabase credentials are not configured.
        Exception:
            If the Supabase client cannot be created.
    """

    global _supabase_client

    # ---------------------------------------------------------
    # 1. Return existing client
    # ---------------------------------------------------------

    if _supabase_client is not None:
        return _supabase_client

    # ---------------------------------------------------------
    # 2. Read Supabase credentials
    # ---------------------------------------------------------

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        logger.error(
            "SUPABASE_URL is not configured"
        )
        raise ValueError(
            "SUPABASE_URL is not configured."
        )

    if not supabase_key:
        logger.error(
            "SUPABASE_KEY is not configured"
        )
        raise ValueError(
            "SUPABASE_KEY is not configured."
        )

    # ---------------------------------------------------------
    # 3. Create Supabase client
    # ---------------------------------------------------------

    try:
        logger.info(
            "Creating Supabase client"
        )

        _supabase_client = create_client(
            supabase_url,
            supabase_key,
        )

        logger.info(
            "Supabase client created successfully"
        )

        return _supabase_client

    except Exception:
        logger.exception(
            "Failed to create Supabase client"
        )
        raise


def check_supabase_health():
    """
    Check the complete Supabase service health.

    Verifies:
        1. Supabase credentials are configured.
        2. Supabase client can be created.
        3. Supabase database is reachable.
        4. The required documents table is accessible.

    Returns:
        dict:
            Structured Supabase health result.
    """

    logger.info(
        "Checking Supabase service health"
    )

    # ---------------------------------------------------------
    # 1. Get Supabase client
    # ---------------------------------------------------------

    try:
        client = get_supabase_client()

    except ValueError as error:
        logger.error(
            "Supabase configuration check failed: %s",
            error,
        )

        return {
            "healthy": False,
            "service": "supabase",
            "database": False,
            "documents_table": False,
            "error": {
                "type": "SUPABASE_CONFIGURATION_ERROR",
                "message": str(error),
            },
        }

    except Exception as error:
        logger.exception(
            "Supabase client health check failed"
        )

        return {
            "healthy": False,
            "service": "supabase",
            "database": False,
            "documents_table": False,
            "error": {
                "type": "SUPABASE_CONNECTION_ERROR",
                "message": str(error),
            },
        }

    # ---------------------------------------------------------
    # 2. Verify database and required table
    # ---------------------------------------------------------

    try:
        logger.info(
            "Checking Supabase table: %s",
            DOCUMENTS_TABLE,
        )

        client.table(
            DOCUMENTS_TABLE
        ).select(
            "id"
        ).limit(
            1
        ).execute()

        logger.info(
            "Supabase service is healthy: "
            "database and '%s' table are accessible",
            DOCUMENTS_TABLE,
        )

        return {
            "healthy": True,
            "service": "supabase",
            "database": True,
            "documents_table": True,
            "error": None,
        }

    except Exception as error:
        logger.exception(
            "Supabase database/table health check failed"
        )

        return {
            "healthy": False,
            "service": "supabase",
            "database": False,
            "documents_table": False,
            "error": {
                "type": "SUPABASE_DATABASE_ERROR",
                "message": str(error),
            },
        }


def reset_supabase_client():
    """
    Reset the shared Supabase client.

    Primarily useful for testing or when the client
    needs to be recreated.
    """
    global _supabase_client
    _supabase_client = None
    logger.info(
        "Supabase client reset"
    )