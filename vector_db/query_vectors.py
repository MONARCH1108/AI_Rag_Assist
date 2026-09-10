import os
from sentence_transformers import SentenceTransformer
from utils.logger import logger

# =============================================================
# EMBEDDING CONFIGURATION
# =============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =============================================================
# SUPABASE CONFIGURATION
# =============================================================

DOCUMENTS_TABLE = "documents"
MATCH_DOCUMENTS_FUNCTION = "match_documents"

# Similarity threshold.
# Lower values return more results.
MATCH_THRESHOLD = 0.20

def query_vectors(
    client,
    query,
    user_id=None,
    documents=None,
    collection_name=DOCUMENTS_TABLE,
    top_k=5,
):
    """
    Search Supabase using a text query.

    The query is embedded using the same Sentence Transformer
    model used during document embedding.

    The generated embedding is passed to the Supabase
    match_documents() PostgreSQL function, which performs
    vector similarity search using pgvector.

    Args:
        client:
            Connected Supabase client.

        query (str):
            User's search query.

        user_id (str):
            Unique identifier of the user/guest whose documents
            should be searched.

        documents (list, optional):
            List of document file names to search within.

            Examples:
                None
                    Search all documents belonging to the user.

                ["document1.pdf"]
                    Search one document belonging to the user.

                ["document1.pdf", "document2.pdf"]
                    Search multiple documents belonging to the user.

        collection_name (str):
            Supabase documents table name.

        top_k (int):
            Number of results to return.

    Returns:
        dict:
            Structured query result containing matching chunks.
    """

    logger.info(
        "Entering Supabase vector query method"
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
            "query": query,
            "results": [],
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
            "User ID was not provided for vector query"
        )
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": {
                "type": "USER_ID_MISSING",
                "message": (
                    "A valid user ID is required for vector search."
                )
            }
        }

    # ---------------------------------------------------------
    # 3. Validate query
    # ---------------------------------------------------------

    if not query or not query.strip():
        logger.warning(
            "Empty query received"
        )
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": {
                "type": "EMPTY_QUERY",
                "message": (
                    "A non-empty query is required."
                )
            }
        }

    # ---------------------------------------------------------
    # 4. Validate top_k
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
                "message": (
                    "top_k must be greater than zero."
                )
            }
        }

    # ---------------------------------------------------------
    # 5. Validate document filter
    # ---------------------------------------------------------

    if documents is not None:
        if not isinstance(documents, list):
            logger.error(
                "Invalid documents parameter. "
                "Expected a list."
            )
            return {
                "success": False,
                "query": query,
                "results": [],
                "error": {
                    "type": "INVALID_DOCUMENT_FILTER",
                    "message": (
                        "documents must be a list of file names "
                        "or None."
                    )
                }
            }

        # Remove empty values
        documents = [
            document.strip()
            for document in documents
            if isinstance(document, str)
            and document.strip()
        ]
        if not documents:
            logger.warning(
                "Document filter was provided but contains "
                "no valid documents."
            )
            return {
                "success": False,
                "query": query,
                "results": [],
                "error": {
                    "type": "EMPTY_DOCUMENT_FILTER",
                    "message": (
                        "At least one valid document is required "
                        "when using a document filter."
                    )
                }
            }
        logger.info(
            "Document filter enabled: %s document(s)",
            len(documents)
        )
        for document in documents:
            logger.info(
                "Document filter: %s",
                document
            )
    else:
        logger.info(
            "No document filter provided. "
            "Searching all documents belonging to user: %s",
            user_id
        )
    try:

        # -----------------------------------------------------
        # 6. Load embedding model
        # -----------------------------------------------------

        logger.info(
            "Loading embedding model for query: %s",
            MODEL_NAME
        )
        hf_token = os.getenv(
            "HF_TOKEN"
        )
        if hf_token:
            model = SentenceTransformer(
                MODEL_NAME,
                token=hf_token
            )
        else:
            model = SentenceTransformer(
                MODEL_NAME
            )
        # -----------------------------------------------------
        # 7. Generate query embedding
        # -----------------------------------------------------

        logger.info(
            "Generating embedding for query"
        )
        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        logger.info(
            "Query embedding generated successfully"
        )

        # -----------------------------------------------------
        # 8. Search Supabase using pgvector RPC
        # -----------------------------------------------------

        logger.info(
            "Searching Supabase table '%s' "
            "for top %s results for user: %s",
            collection_name,
            top_k,
            user_id
        )
        response = client.rpc(
            MATCH_DOCUMENTS_FUNCTION,
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count": top_k,
                "filter_documents": documents,
                "filter_user_id": user_id,
            }
        ).execute()
        rows = response.data or []
        logger.info(
            "Supabase returned %s matching chunks",
            len(rows)
        )

        # -----------------------------------------------------
        # 9. Format search results
        # -----------------------------------------------------

        results = []
        for row in rows:
            results.append(
                {
                    "score": row.get(
                        "similarity",
                        0
                    ),
                    "page_content": row.get(
                        "page_content",
                        ""
                    ),
                    "metadata": row.get(
                        "metadata",
                        {}
                    ),
                    "file_name": row.get(
                        "file_name"
                    ),
                }
            )
        logger.info(
            "Supabase vector query completed successfully: "
            "%s results returned",
            len(results)
        )

        # -----------------------------------------------------
        # 10. Return successful result
        # -----------------------------------------------------

        return {
            "success": True,
            "query": query,
            "documents": documents,
            "results": results,
            "error": None,
        }
    except Exception as error:
        logger.exception(
            "Supabase vector query failed"
        )
        return {
            "success": False,
            "query": query,
            "documents": documents,
            "results": [],
            "error": {
                "type": "SUPABASE_QUERY_ERROR",
                "message": str(error),
            }
        }