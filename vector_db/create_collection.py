import logging
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)

def create_qdrant_collection(
    client,
    collection_name="ai_rag_documents",
    vector_size=384,
):
    """
    Create a Qdrant collection if it does not already exist.

    Args:
        client: Connected QdrantClient instance.
        collection_name (str): Qdrant collection name.
        vector_size (int): Dimension of the embedding vectors.

    Returns:
        dict: Structured collection creation result.
    """

    logger.info(
        "Entering Qdrant collection creation method"
    )

    # ---------------------------------------------------------
    # 1. Validate client
    # ---------------------------------------------------------

    if client is None:
        logger.error("Qdrant client was not provided")
        return {
            "success": False,
            "collection_name": collection_name,
            "created": False,
            "error": {
                "type": "QDRANT_CLIENT_MISSING",
                "message": "A valid Qdrant client is required."
            }
        }

    # ---------------------------------------------------------
    # 2. Check whether collection already exists
    # ---------------------------------------------------------

    try:
        collections = client.get_collections()
        collection_exists = any(
            collection.name == collection_name
            for collection in collections.collections
        )
        if collection_exists:
            logger.info(
                "Qdrant collection already exists: %s",
                collection_name
            )
            return {
                "success": True,
                "collection_name": collection_name,
                "created": False,
                "error": None
            }

    except Exception as error:
        logger.exception(
            "Failed to check Qdrant collection: %s",
            collection_name
        )

        return {
            "success": False,
            "collection_name": collection_name,
            "created": False,
            "error": {
                "type": "COLLECTION_CHECK_ERROR",
                "message": str(error)
            }
        }

    # ---------------------------------------------------------
    # 3. Create collection
    # ---------------------------------------------------------

    try:
        logger.info(
            "Creating Qdrant collection: %s",
            collection_name
        )

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            "Successfully created Qdrant collection: %s",
            collection_name
        )

        return {
            "success": True,
            "collection_name": collection_name,
            "created": True,
            "error": None
        }

    except Exception as error:
        logger.exception(
            "Failed to create Qdrant collection: %s",
            collection_name
        )

        return {
            "success": False,
            "collection_name": collection_name,
            "created": False,
            "error": {
                "type": "COLLECTION_CREATION_ERROR",
                "message": str(error)
            }
        }

if __name__ == "__main__":
    from qdrant import connect_to_qdrant
    connection = connect_to_qdrant()
    if not connection["success"]:
        print(connection)
    else:
        response = create_qdrant_collection(
            connection["client"]
        )
        print(response)