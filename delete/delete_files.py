import logging

logger = logging.getLogger(__name__)

DOCUMENTS_TABLE = "documents"

def delete_document(
    client,
    file_name,
    table_name=DOCUMENTS_TABLE,
):
    """
    Delete all chunks belonging to a single document
    from the Supabase documents table.

    Args:
        client:
            Connected Supabase client.

        file_name (str):
            Name of the document to delete.

        table_name (str):
            Supabase documents table.

    Returns:
        dict:
            Structured deletion result.
    """

    logger.info(
        "Starting deletion of document: %s",
        file_name,
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
            "deleted_count": 0,
            "file_name": file_name,
            "error": {
                "type": "SUPABASE_CLIENT_MISSING",
                "message": "A valid Supabase client is required.",
            },
        }

    # ---------------------------------------------------------
    # 2. Validate file name
    # ---------------------------------------------------------

    if not file_name or not file_name.strip():
        logger.warning(
            "Empty file name received for deletion"
        )
        return {
            "success": False,
            "deleted_count": 0,
            "file_name": file_name,
            "error": {
                "type": "EMPTY_FILE_NAME",
                "message": "A valid file name is required.",
            },
        }
    file_name = file_name.strip()

    # ---------------------------------------------------------
    # 3. Delete all chunks belonging to the document
    # ---------------------------------------------------------
    try:
        logger.info(
            "Deleting all chunks for document: %s",
            file_name,
        )
        response = (
            client
            .table(table_name)
            .delete()
            .eq("file_name", file_name)
            .execute()
        )
        deleted_rows = response.data or []
        deleted_count = len(deleted_rows)
        logger.info(
            "Successfully deleted %s rows for document: %s",
            deleted_count,
            file_name,
        )
        return {
            "success": True,
            "deleted_count": deleted_count,
            "file_name": file_name,
            "error": None,
        }
    except Exception as error:
        logger.exception(
            "Failed to delete document: %s",
            file_name,
        )
        return {
            "success": False,
            "deleted_count": 0,
            "file_name": file_name,
            "error": {
                "type": "DOCUMENT_DELETION_ERROR",
                "message": str(error),
            },
        }

def delete_all_documents(
    client,
    table_name=DOCUMENTS_TABLE,
):
    """
    Delete all document chunks from the Supabase
    documents table.

    Args:
        client:
            Connected Supabase client.

        table_name (str):
            Supabase documents table.

    Returns:
        dict:
            Structured deletion result.
    """

    logger.info(
        "Starting deletion of all documents"
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
            "deleted_count": 0,
            "error": {
                "type": "SUPABASE_CLIENT_MISSING",
                "message": "A valid Supabase client is required.",
            },
        }

    # ---------------------------------------------------------
    # 2. Delete all rows
    # ---------------------------------------------------------

    try:
        logger.warning(
            "Deleting ALL documents from Supabase"
        )
        response = (
            client
            .table(table_name)
            .delete()
            .neq("id", 0)
            .execute()
        )
        deleted_rows = response.data or []
        deleted_count = len(deleted_rows)
        logger.info(
            "Successfully deleted %s rows from Supabase",
            deleted_count,
        )
        return {
            "success": True,
            "deleted_count": deleted_count,
            "error": None,
        }
    except Exception as error:
        logger.exception(
            "Failed to delete all documents"
        )
        return {
            "success": False,
            "deleted_count": 0,
            "error": {
                "type": "ALL_DOCUMENTS_DELETION_ERROR",
                "message": str(error),
            },
        }