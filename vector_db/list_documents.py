import logging
logger = logging.getLogger(__name__)

# =============================================================
# SUPABASE DATABASE CONFIGURATION
# =============================================================
DOCUMENTS_TABLE = "documents"
def list_documents(
    client,
    table_name=DOCUMENTS_TABLE,
):
    """
    List the unique documents currently stored in Supabase.

    Args:
        client:
            Connected Supabase client.

        table_name (str):
            Supabase documents table name.

    Returns:
        dict:
            Structured result containing the existing documents.
    """

    logger.info(
        "Starting document listing from Supabase table: %s",
        table_name
    )

    # ---------------------------------------------------------
    # 1. Validate Supabase client
    # ---------------------------------------------------------
    if client is None:
        logger.error(
            "Supabase client was not provided"
        )
        return {
            "success": False,
            "documents": [],
            "error": {
                "type": "SUPABASE_CLIENT_MISSING",
                "message": (
                    "A valid Supabase client is required."
                )
            }
        }
    try:
        # -----------------------------------------------------
        # 2. Retrieve stored documents
        # -----------------------------------------------------
        response = (
            client
            .table(table_name)
            .select(
                "file_name, metadata"
            )
            .execute()
        )
        rows = response.data or []
        logger.info(
            "Retrieved %s records from Supabase",
            len(rows)
        )

        # -----------------------------------------------------
        # 3. Extract unique documents
        # -----------------------------------------------------
        documents = {}
        for row in rows:
            file_name = row.get(
                "file_name"
            )
            metadata = row.get(
                "metadata"
            ) or {}
            if not file_name:
                continue
            file_type = metadata.get(
                "file_type"
            )
            source = metadata.get(
                "source"
            )
            document_key = (
                source
                or file_name
            )
            if document_key not in documents:
                documents[document_key] = {
                    "file_name": file_name,
                    "file_type": file_type,
                    "source": source,
                }
        document_list = list(
            documents.values()
        )
        logger.info(
            "Found %s unique documents in Supabase",
            len(document_list)
        )

        # -----------------------------------------------------
        # 4. Return successful result
        # -----------------------------------------------------

        return {
            "success": True,
            "documents": document_list,
            "error": None,
        }
    except Exception as error:
        logger.exception(
            "Failed to list documents from Supabase"
        )
        return {
            "success": False,
            "documents": [],
            "error": {
                "type": "DOCUMENT_LISTING_ERROR",
                "message": str(error),
            }
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    from supabase_client import (
        connect_to_supabase
    )
    load_dotenv()
    connection = connect_to_supabase()
    if not connection["success"]:
        print(connection)
    else:
        response = list_documents(
            client=connection["client"]
        )
        print(response)