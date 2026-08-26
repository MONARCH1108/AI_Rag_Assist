import logging
import os
import time
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# =============================================================
# SUPABASE CONFIGURATION
# =============================================================

SUPABASE_MAX_RETRIES = 3
SUPABASE_RETRY_DELAY = 2
def connect_to_supabase():
    """
    Connect to the Supabase project and verify the connection.

    Returns:
        dict: Structured Supabase connection result.
    """

    # ---------------------------------------------------------
    # 1. Read Supabase credentials
    # ---------------------------------------------------------
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url:
        logger.error(
            "SUPABASE_URL is not configured"
        )
        return {
            "success": False,
            "client": None,
            "error": {
                "type": "SUPABASE_URL_MISSING",
                "message": "SUPABASE_URL is not configured."
            }
        }

    if not supabase_key:
        logger.error(
            "SUPABASE_KEY is not configured"
        )
        return {
            "success": False,
            "client": None,
            "error": {
                "type": "SUPABASE_KEY_MISSING",
                "message": "SUPABASE_KEY is not configured."
            }
        }

    # ---------------------------------------------------------
    # 2. Create Supabase client
    # ---------------------------------------------------------

    for attempt in range(
        1,
        SUPABASE_MAX_RETRIES + 1
    ):
        try:
            logger.info(
                "Connecting to Supabase (attempt %s/%s)",
                attempt,
                SUPABASE_MAX_RETRIES
            )
            client: Client = create_client(
                supabase_url,
                supabase_key
            )

            # -------------------------------------------------
            # 3. Verify connection
            # -------------------------------------------------

            # Query the documents table to verify that the
            # Supabase API and database are reachable.
            client.table(
                "documents"
            ).select(
                "id"
            ).limit(
                1
            ).execute()

            logger.info(
                "Successfully connected to Supabase"
            )
            return {
                "success": True,
                "client": client,
                "error": None
            }

        except Exception as error:
            logger.warning(
                "Supabase connection attempt %s/%s failed: %s",
                attempt,
                SUPABASE_MAX_RETRIES,
                error
            )

            # -------------------------------------------------
            # 4. Retry with exponential backoff
            # -------------------------------------------------

            if attempt < SUPABASE_MAX_RETRIES:
                wait_time = (
                    SUPABASE_RETRY_DELAY
                    * (2 ** (attempt - 1))
                )
                logger.info(
                    "Retrying Supabase connection in %s seconds",
                    wait_time
                )
                time.sleep(wait_time)

    # ---------------------------------------------------------
    # 5. All attempts failed
    # ---------------------------------------------------------

    logger.error(
        "Unable to connect to Supabase after %s attempts",
        SUPABASE_MAX_RETRIES
    )
    return {
        "success": False,
        "client": None,
        "error": {
            "type": "SUPABASE_CONNECTION_ERROR",
            "message": (
                "Unable to connect to Supabase "
                f"after {SUPABASE_MAX_RETRIES} attempts."
            )
        }
    }


if __name__ == "__main__":
    response = connect_to_supabase()
    print(response)