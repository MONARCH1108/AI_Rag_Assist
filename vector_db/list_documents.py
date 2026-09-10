from utils.logger import logger


# =============================================================
# SUPABASE DATABASE CONFIGURATION
# =============================================================

DOCUMENTS_TABLE = "documents"
def list_documents(
    client,
    user_id=None,
    table_name=DOCUMENTS_TABLE,
):
    """
    List the unique documents belonging to a specific user
    currently stored in Supabase.

    Args:
        client:
            Connected Supabase client.

        user_id (str):
            Unique identifier of the user/guest whose documents
            should be listed.

        table_name (str):
            Supabase documents table name.

    Returns:
        dict:
            Structured result containing the user's documents.
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

    # ---------------------------------------------------------
    # 2. Validate user ID
    # ---------------------------------------------------------

    if not user_id:
        logger.error(
            "User ID was not provided for document listing"
        )
        return {
            "success": False,
            "documents": [],
            "error": {
                "type": "USER_ID_MISSING",
                "message": (
                    "A valid user ID is required to list documents."
                )
            }
        }

    try:
        # -----------------------------------------------------
        # 3. Retrieve stored documents belonging to the user
        # -----------------------------------------------------

        response = (
            client
            .table(table_name)
            .select(
                "file_name, metadata"
            )
            .eq(
                "user_id",
                user_id
            )
            .execute()
        )
        rows = response.data or []
        logger.info(
            "Retrieved %s records from Supabase for user: %s",
            len(rows),
            user_id
        )

        # -----------------------------------------------------
        # 4. Extract unique documents
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
            "Found %s unique documents for user: %s",
            len(document_list),
            user_id
        )

        # -----------------------------------------------------
        # 5. Return successful result
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
        user_id = input(
            "Enter user ID: "
        ).strip()

        response = list_documents(
            client=connection["client"],
            user_id=user_id,
        )

        print(response)