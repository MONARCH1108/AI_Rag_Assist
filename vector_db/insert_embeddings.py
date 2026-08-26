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
    # 6. Insert vectors into Qdrant
    # ---------------------------------------------------------

    try:
        logger.info(
            "Inserting %s vectors into Qdrant collection: %s",
            len(points),
            collection_name,
        )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        logger.info(
            "Successfully inserted %s vectors into Qdrant collection: %s",
            len(points),
            collection_name,
        )

        # -----------------------------------------------------
        # 7. Return successful result
        # -----------------------------------------------------

        return {
            "success": True,
            "collection_name": collection_name,
            "inserted_count": len(points),
            "error": None,
        }

    except Exception as error:
        logger.exception(
            "Failed to insert vectors into Qdrant collection: %s",
            collection_name,
        )

        return {
            "success": False,
            "collection_name": collection_name,
            "inserted_count": 0,
            "error": {
                "type": "QDRANT_INSERTION_ERROR",
                "message": str(error),
            }
        }