import logging
logger = logging.getLogger(__name__)


# =============================================================
# SUPABASE DATABASE CONFIGURATION
# =============================================================

DOCUMENTS_TABLE = "documents"
def verify_supabase_database(
    client,
    table_name=DOCUMENTS_TABLE,
):
    """
    Verify that the required Supabase database table exists
    and is accessible.

    The database schema, vector column, HNSW index, and
    match_documents() function are created through the
    Supabase SQL editor and are not recreated at runtime.

    Args:
        client:
            Connected Supabase client.

        table_name (str):
            Name of the Supabase documents table.

    Returns:
        dict:
            Structured database verification result.
    """

    logger.info(
        "Verifying Supabase database"
    )
    # ---------------------------------------------------------
    # 1. Validate client
    # ---------------------------------------------------------

    if client is None:
        logger.error(
            "Supabase client was not provided"
        )
        return {
            "success": False,
            "table_name": table_name,
            "created": False,
            "error": {
                "type": "SUPABASE_CLIENT_MISSING",
                "message": (
                    "A valid Supabase client is required."
                )
            }
        }

    # ---------------------------------------------------------
    # 2. Verify documents table
    # ---------------------------------------------------------

    try:

        client.table(
            table_name
        ).select(
            "id"
        ).limit(
            1
        ).execute()

        logger.info(
            "Supabase documents table is available: %s",
            table_name
        )
        return {
            "success": True,
            "table_name": table_name,
            "created": False,
            "error": None
        }
    except Exception as error:
        logger.exception(
            "Failed to verify Supabase documents table: %s",
            table_name
        )
        return {
            "success": False,
            "table_name": table_name,
            "created": False,
            "error": {
                "type": "DATABASE_VERIFICATION_ERROR",
                "message": str(error)
            }
        }

if __name__ == "__main__":
    from supabase_client import connect_to_supabase
    from dotenv import load_dotenv

    load_dotenv()
    connection = connect_to_supabase()
    if not connection["success"]:
        print(connection)

    else:
        response = verify_supabase_database(
            connection["client"]
        )
        print(response)