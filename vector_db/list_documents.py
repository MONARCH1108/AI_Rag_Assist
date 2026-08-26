import logging

logger = logging.getLogger(__name__)

def list_documents(
    client,
    collection_name="ai_rag_documents",
):
    """
    List the unique documents currently stored in Qdrant.

    Args:
        client: Connected QdrantClient instance.
        collection_name (str): Qdrant collection name.

    Returns:
        dict: Structured result containing the existing documents.
    """
    logger.info(
        "Starting document listing from Qdrant collection: %s",
        collection_name
    )

    # ---------------------------------------------------------
    # 1. Validate Qdrant client
    # ---------------------------------------------------------
    if client is None:
        logger.error("Qdrant client was not provided")
        return {
            "success": False,
            "documents": [],
            "error": {
                "type": "QDRANT_CLIENT_MISSING",
                "message": "A valid Qdrant client is required."
            }
        }

    try:
        # -----------------------------------------------------
        # 2. Retrieve stored points
        # -----------------------------------------------------
        points = []
        offset = None
        while True:
            response = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            batch, next_offset = response
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        logger.info(
            "Retrieved %s points from Qdrant",
            len(points)
        )

        # -----------------------------------------------------
        # 3. Extract unique documents
        # -----------------------------------------------------
        documents = {}

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            file_name = metadata.get("file_name")
            file_type = metadata.get("file_type")
            source = metadata.get("source")
            if not file_name:
                continue

            document_key = source or file_name
            if document_key not in documents:
                documents[document_key] = {
                    "file_name": file_name,
                    "file_type": file_type,
                    "source": source,
                }

        document_list = list(documents.values())
        logger.info(
            "Found %s unique documents in Qdrant",
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
            "Failed to list documents from Qdrant"
        )
        return {
            "success": False,
            "documents": [],
            "error": {
                "type": "DOCUMENT_LISTING_ERROR",
                "message": str(error),
            }
        }