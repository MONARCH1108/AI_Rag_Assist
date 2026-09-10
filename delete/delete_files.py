from utils.logger import logger


DOCUMENTS_TABLE = "documents"


def delete_document(
    client,
    user_id,
    file_name,
    table_name=DOCUMENTS_TABLE,
):
    """
    Delete all chunks belonging to a single document
    for a specific user from the Supabase documents table.

    Args:
        client:
            Connected Supabase client.

        user_id (str):
            Unique user/guest ID owning the document.

        file_name (str):
            Name of the document to delete.

        table_name (str):
            Supabase documents table.

    Returns:
        dict:
            Structured deletion result.
    """

    logger.info(
        "Starting deletion of document: %s for user: %s",
        file_name,
        user_id,
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
    # 2. Validate user ID
    # ---------------------------------------------------------

    if not user_id or not str(user_id).strip():
        logger.warning(
            "Missing user ID for document deletion"
        )
        return {
            "success": False,
            "deleted_count": 0,
            "file_name": file_name,
            "error": {
                "type": "USER_ID_MISSING",
                "message": "A valid user ID is required.",
            },
        }

    user_id = str(user_id).strip()

    # ---------------------------------------------------------
    # 3. Validate file name
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
    # 4. Delete all chunks belonging to the document
    #    for the specified user
    # ---------------------------------------------------------

    try:
        logger.info(
            "Deleting document: %s for user: %s",
            file_name,
            user_id,
        )

        response = (
            client
            .table(table_name)
            .delete()
            .eq("user_id", user_id)
            .eq("file_name", file_name)
            .execute()
        )

        deleted_rows = response.data or []
        deleted_count = len(deleted_rows)

        logger.info(
            "Successfully deleted %s rows for document: %s "
            "for user: %s",
            deleted_count,
            file_name,
            user_id,
        )

        return {
            "success": True,
            "deleted_count": deleted_count,
            "file_name": file_name,
            "error": None,
        }

    except Exception as error:
        logger.exception(
            "Failed to delete document: %s for user: %s",
            file_name,
            user_id,
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
    user_id,
    table_name=DOCUMENTS_TABLE,
):
    """
    Delete all document chunks belonging to a specific user
    from the Supabase documents table.

    Args:
        client:
            Connected Supabase client.

        user_id (str):
            Unique user/guest ID owning the documents.

        table_name (str):
            Supabase documents table.

    Returns:
        dict:
            Structured deletion result.
    """

    logger.info(
        "Starting deletion of all documents for user: %s",
        user_id,
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
    # 2. Validate user ID
    # ---------------------------------------------------------

    if not user_id or not str(user_id).strip():
        logger.warning(
            "Missing user ID for deleting all documents"
        )

        return {
            "success": False,
            "deleted_count": 0,
            "error": {
                "type": "USER_ID_MISSING",
                "message": "A valid user ID is required.",
            },
        }

    user_id = str(user_id).strip()

    # ---------------------------------------------------------
    # 3. Delete all documents belonging to the user
    # ---------------------------------------------------------

    try:
        logger.warning(
            "Deleting ALL documents for user: %s",
            user_id,
        )

        response = (
            client
            .table(table_name)
            .delete()
            .eq("user_id", user_id)
            .execute()
        )

        deleted_rows = response.data or []
        deleted_count = len(deleted_rows)

        logger.info(
            "Successfully deleted %s rows for user: %s",
            deleted_count,
            user_id,
        )

        return {
            "success": True,
            "deleted_count": deleted_count,
            "error": None,
        }

    except Exception as error:
        logger.exception(
            "Failed to delete all documents for user: %s",
            user_id,
        )

        return {
            "success": False,
            "deleted_count": 0,
            "error": {
                "type": "ALL_DOCUMENTS_DELETION_ERROR",
                "message": str(error),
            },
        }