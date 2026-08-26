import logging
import os
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)

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
    collection_name=DOCUMENTS_TABLE,
    top_k=5,
):
    """
    Search Supabase using a text query.

    The query is embedded using the same Sentence Transformer
    model used during document embedding.

    The generated embedding is then passed to the Supabase
    match_documents() PostgreSQL function, which performs
    vector similarity search using pgvector.

    Args:
        client:
            Connected Supabase client.

        query (str):
            User's search query.

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
    # 2. Validate query
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
                "message": (
                    "top_k must be greater than zero."
                )
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
        hf_token = os.getenv("HF_TOKEN")
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
        # 5. Generate query embedding
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
        # 6. Search Supabase using pgvector RPC
        # -----------------------------------------------------

        logger.info(
            "Searching Supabase table '%s' for top %s results",
            collection_name,
            top_k
        )
        response = client.rpc(
            MATCH_DOCUMENTS_FUNCTION,
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count": top_k,
            }
        ).execute()
        rows = response.data or []

        # -----------------------------------------------------
        # 7. Format search results
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
            "Supabase vector query failed"
        )

        return {
            "success": False,
            "query": query,
            "results": [],
            "error": {
                "type": "SUPABASE_QUERY_ERROR",
                "message": str(error),
            }
        }


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    from vector_db.supabase_client import (
        connect_to_supabase
    )
    load_dotenv()
    query = input(
        "Enter your query: "
    ).strip()
    connection = connect_to_supabase()
    if not connection["success"]:
        print(connection)
    else:
        response = query_vectors(
            client=connection["client"],
            query=query,
            top_k=5,
        )

        print(response)