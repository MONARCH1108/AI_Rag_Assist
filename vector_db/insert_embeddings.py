import logging
import uuid
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


def insert_embeddings(
    client,
    chunks,
    embeddings,
    collection_name="ai_rag_documents",
):
    """
    Insert embedded document chunks into a Qdrant collection.

    Args:
        client: Connected QdrantClient instance.
        chunks (list): List of LangChain Document objects.
        embeddings (list): Embedding vectors corresponding to each chunk.
        collection_name (str): Name of the Qdrant collection.

    Returns:
        dict: Structured vector insertion result.
    """

    logger.info(
        "Entering Qdrant embedding insertion method"
    )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    BATCH_SIZE = 25

    # ---------------------------------------------------------
    # 1. Validate chunks
    # ---------------------------------------------------------

    if not chunks:
        logger.warning(
            "No chunks were provided for Qdrant insertion"
        )

        return {
            "success": False,
            "collection_name": collection_name,
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
            "No embeddings were provided for Qdrant insertion"
        )

        return {
            "success": False,
            "collection_name": collection_name,
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
            "Chunk and embedding count mismatch: %s chunks, %s embeddings",
            len(chunks),
            len(embeddings),
        )

        return {
            "success": False,
            "collection_name": collection_name,
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
    # 4. Validate Qdrant client
    # ---------------------------------------------------------

    if client is None:
        logger.error(
            "Qdrant client was not provided"
        )

        return {
            "success": False,
            "collection_name": collection_name,
            "inserted_count": 0,
            "error": {
                "type": "QDRANT_CLIENT_MISSING",
                "message": "A valid Qdrant client is required."
            }
        }

    # ---------------------------------------------------------
    # 5. Prepare Qdrant points
    # ---------------------------------------------------------

    try:
        logger.info(
            "Preparing %s vectors for Qdrant collection: %s",
            len(chunks),
            collection_name,
        )

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            point_id = str(uuid.uuid4())

            payload = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )

            points.append(point)

        logger.info(
            "Successfully prepared %s vectors",
            len(points),
        )

    except Exception as error:
        logger.exception(
            "Failed to prepare vectors for Qdrant insertion"
        )

        return {
            "success": False,
            "collection_name": collection_name,
            "inserted_count": 0,
            "error": {
                "type": "POINT_PREPARATION_ERROR",
                "message": str(error),
            }
        }

    # ---------------------------------------------------------
    # 6. Insert vectors into Qdrant in batches
    # ---------------------------------------------------------

    total_points = len(points)
    total_batches = (total_points + BATCH_SIZE - 1) // BATCH_SIZE
    inserted_count = 0

    logger.info(
        "Starting batched Qdrant insertion: %s vectors, "
        "batch size: %s, total batches: %s",
        total_points,
        BATCH_SIZE,
        total_batches,
    )

    try:

        for batch_start in range(0, total_points, BATCH_SIZE):

            batch_end = min(
                batch_start + BATCH_SIZE,
                total_points,
            )

            batch = points[batch_start:batch_end]

            batch_number = (batch_start // BATCH_SIZE) + 1

            logger.info(
                "Inserting batch %s/%s: %s vectors",
                batch_number,
                total_batches,
                len(batch),
            )

            client.upsert(
                collection_name=collection_name,
                points=batch,
            )

            inserted_count += len(batch)

            logger.info(
                "Successfully inserted batch %s/%s: "
                "%s vectors",
                batch_number,
                total_batches,
                len(batch),
            )

        # -----------------------------------------------------
        # 7. Return successful result
        # -----------------------------------------------------

        logger.info(
            "Successfully inserted all %s vectors into "
            "Qdrant collection: %s",
            inserted_count,
            collection_name,
        )

        return {
            "success": True,
            "collection_name": collection_name,
            "inserted_count": inserted_count,
            "error": None,
        }

    except Exception as error:

        logger.exception(
            "Failed during Qdrant batch insertion. "
            "Successfully inserted %s/%s vectors into "
            "collection: %s",
            inserted_count,
            total_points,
            collection_name,
        )

        return {
            "success": False,
            "collection_name": collection_name,
            "inserted_count": inserted_count,
            "error": {
                "type": "QDRANT_INSERTION_ERROR",
                "message": str(error),
            }
        }