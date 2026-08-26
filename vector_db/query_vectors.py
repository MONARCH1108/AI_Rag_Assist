import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def query_vectors(
    client,
    query,
    collection_name="ai_rag_documents",
    top_k=5,
):
    """
    Search Qdrant using a text query.
    The query is embedded using the same Sentence Transformer
    model used during document embedding.
    Args:
        client: Connected QdrantClient instance.
        query (str): User's search query.
        collection_name (str): Qdrant collection name.
        top_k (int): Number of results to return.
    Returns:
        dict: Structured query result containing matching chunks.
    """
    logger.info("Entering Qdrant vector query method")

    # ---------------------------------------------------------
    # 1. Validate Qdrant client
    # ---------------------------------------------------------

    if client is None:
        logger.error("Qdrant client was not provided")
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": {
                "type": "QDRANT_CLIENT_MISSING",
                "message": "A valid Qdrant client is required."
            }
        }

    # ---------------------------------------------------------
    # 2. Validate query
    # ---------------------------------------------------------

    if not query or not query.strip():
        logger.warning("Empty query received")
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": {
                "type": "EMPTY_QUERY",
                "message": "A non-empty query is required."
            }
        }

    # ---------------------------------------------------------
    # 3. Validate top_k
    # ---------------------------------------------------------

    if top_k <= 0:
        logger.warning(
            "Invalid top_k value: %s",
            top_k
        )
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": {
                "type": "INVALID_TOP_K",
                "message": "top_k must be greater than zero."
            }
        }
    try:
        # -----------------------------------------------------
        # 4. Load embedding model
        # -----------------------------------------------------

        logger.info(
            "Loading embedding model for query: %s",
            MODEL_NAME
        )
        model = SentenceTransformer(MODEL_NAME)
        # -----------------------------------------------------
        # 5. Generate query embedding
        # -----------------------------------------------------

        logger.info("Generating embedding for query")
        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        logger.info(
            "Query embedding generated successfully"
        )

        # -----------------------------------------------------
        # 6. Search Qdrant
        # -----------------------------------------------------

        logger.info(
            "Searching Qdrant collection '%s' for top %s results",
            collection_name,
            top_k,
        )
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        points = search_result.points

        # -----------------------------------------------------
        # 7. Format search results
        # -----------------------------------------------------

        results = []
        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "score": point.score,
                    "page_content": payload.get("page_content", ""),
                    "metadata": payload.get("metadata", {}),
                }
            )
        logger.info(
            "Qdrant query completed successfully: %s results returned",
            len(results),
        )

        # -----------------------------------------------------
        # 8. Return successful result
        # -----------------------------------------------------

        return {
            "success": True,
            "query": query,
            "results": results,
            "error": None,
        }
    except Exception as error:
        logger.exception(
            "Qdrant vector query failed"
        )
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": {
                "type": "QDRANT_QUERY_ERROR",
                "message": str(error),
            }
        }


#if __name__ == "__main__":
#    from qdrant import connect_to_qdrant
#    load_dotenv()
#    query = input("Enter your query: ").strip()
#    connection = connect_to_qdrant()
#    if not connection["success"]:
#        print(connection)
#    else:
#        response = query_vectors(
#            client=connection["client"],
#            query=query,
#            top_k=5,
#        )
#        print(response)