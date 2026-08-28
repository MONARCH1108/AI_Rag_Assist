from utils.logger import logger


# =============================================================
# SUPABASE DATABASE CONFIGURATION
# =============================================================

DOCUMENTS_TABLE = "documents"
BATCH_SIZE = 25


def insert_documents(
    client,
    chunks,
    embeddings,
    table_name=DOCUMENTS_TABLE,
):
    """
    Insert embedded document chunks into the Supabase
    documents table.

    Args:
        client:
            Connected Supabase client.

        chunks (list):
            List of LangChain Document objects.

        embeddings (list):
            Embedding vectors corresponding to each chunk.

        table_name (str):
            Supabase table name.

    Returns:
        dict:
            Structured document insertion result.
    """

    logger.info(
        "Entering Supabase document insertion method"
    )

    # ---------------------------------------------------------
    # 1. Validate chunks
    # ---------------------------------------------------------

    if not chunks:
        logger.warning(
            "No chunks were provided for Supabase insertion"
        )

        return {
            "success": False,
            "table_name": table_name,
            "inserted_count": 0,
            "error": {
                "type": "EMPTY_CHUNKS",
                "message": "No document chunks were provided."
            }
        }

    # ---------------------------------------------------------
    # 2. Validate embeddings
    # ---------------------------------------------------------

    if not embeddings:
        logger.warning(
            "No embeddings were provided for Supabase insertion"
        )

        return {
            "success": False,
            "table_name": table_name,
            "inserted_count": 0,
            "error": {
                "type": "EMPTY_EMBEDDINGS",
                "message": "No embeddings were provided."
            }
        }

    # ---------------------------------------------------------
    # 3. Make sure chunks and embeddings match
    # ---------------------------------------------------------

    if len(chunks) != len(embeddings):
        logger.error(
            "Chunk and embedding count mismatch: "
            "%s chunks, %s embeddings",
            len(chunks),
            len(embeddings),
        )

        return {
            "success": False,
            "table_name": table_name,
            "inserted_count": 0,
            "error": {
                "type": "COUNT_MISMATCH",
                "message": (
                    f"Number of chunks ({len(chunks)}) does not match "
                    f"number of embeddings ({len(embeddings)})."
                )
            }
        }

    # ---------------------------------------------------------
    # 4. Validate Supabase client
    # ---------------------------------------------------------

    if client is None:
        logger.error(
            "Supabase client was not provided"
        )

        return {
            "success": False,
            "table_name": table_name,
            "inserted_count": 0,
            "error": {
                "type": "SUPABASE_CLIENT_MISSING",
                "message": (
                    "A valid Supabase client is required."
                )
            }
        }

    # ---------------------------------------------------------
    # 5. Prepare Supabase records
    # ---------------------------------------------------------

    try:
        logger.info(
            "Preparing %s documents for Supabase table: %s",
            len(chunks),
            table_name,
        )

        records = []

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):
            metadata = chunk.metadata or {}

            file_name = metadata.get(
                "file_name",
                ""
            )

            record = {
                "file_name": file_name,
                "page_content": chunk.page_content,
                "metadata": metadata,
                "embedding": embedding,
            }

            records.append(record)

        logger.info(
            "Successfully prepared %s documents",
            len(records),
        )

    except Exception as error:
        logger.exception(
            "Failed to prepare documents for Supabase insertion"
        )

        return {
            "success": False,
            "table_name": table_name,
            "inserted_count": 0,
            "error": {
                "type": "DOCUMENT_PREPARATION_ERROR",
                "message": str(error),
            }
        }

    # ---------------------------------------------------------
    # 6. Insert documents into Supabase in batches
    # ---------------------------------------------------------

    total_records = len(records)

    total_batches = (
        (total_records + BATCH_SIZE - 1)
        // BATCH_SIZE
    )

    inserted_count = 0

    logger.info(
        "Starting batched Supabase insertion: "
        "%s documents, batch size: %s, total batches: %s",
        total_records,
        BATCH_SIZE,
        total_batches,
    )

    try:
        for batch_start in range(
            0,
            total_records,
            BATCH_SIZE
        ):
            batch_end = min(
                batch_start + BATCH_SIZE,
                total_records,
            )

            batch = records[
                batch_start:batch_end
            ]

            batch_number = (
                batch_start // BATCH_SIZE
            ) + 1

            logger.info(
                "Inserting batch %s/%s: %s documents",
                batch_number,
                total_batches,
                len(batch),
            )

            response = (
                client
                .table(table_name)
                .insert(batch)
                .execute()
            )

            # -------------------------------------------------
            # Verify Supabase returned the inserted records
            # -------------------------------------------------

            if response.data is None:
                raise RuntimeError(
                    "Supabase did not return inserted records."
                )

            inserted_batch_count = len(
                response.data
            )

            inserted_count += (
                inserted_batch_count
            )

            logger.info(
                "Successfully inserted batch %s/%s: "
                "%s documents",
                batch_number,
                total_batches,
                inserted_batch_count,
            )

        # -----------------------------------------------------
        # 7. Return successful result
        # -----------------------------------------------------

        logger.info(
            "Successfully inserted all %s documents "
            "into Supabase table: %s",
            inserted_count,
            table_name,
        )

        return {
            "success": True,
            "table_name": table_name,
            "inserted_count": inserted_count,
            "error": None,
        }

    except Exception as error:
        logger.exception(
            "Failed during Supabase batch insertion. "
            "Successfully inserted %s/%s documents "
            "into table: %s",
            inserted_count,
            total_records,
            table_name,
        )

        return {
            "success": False,
            "table_name": table_name,
            "inserted_count": inserted_count,
            "error": {
                "type": "SUPABASE_INSERTION_ERROR",
                "message": str(error),
            }
        }