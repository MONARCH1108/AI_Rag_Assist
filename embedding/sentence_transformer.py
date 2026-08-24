import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =============================================================
# EMBEDDING CONFIGURATION
# =============================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_chunks(
    chunks,
    model_name=EMBEDDING_MODEL_NAME
):
    """
    Generate vector embeddings for LangChain document chunks.

    Args:
        chunks (list): List of LangChain Document objects.
        model_name (str): Hugging Face Sentence Transformer model name.

    Returns:
        dict: Structured embedding result.
    """

    logger.info(
        "Entering sentence transformer embedding method"
    )
    # ---------------------------------------------------------
    # 1. Validate chunks
    # ---------------------------------------------------------
    if not chunks:
        logger.warning(
            "No chunks were provided for embedding"
        )
        return {
            "success": False,
            "model_name": model_name,
            "total_chunks": 0,
            "embedded_chunks": 0,
            "embeddings": [],
            "error": {
                "type": "EMPTY_CHUNKS",
                "message": "No chunks were provided for embedding."
            }
        }

    # ---------------------------------------------------------
    # 2. Load embedding model
    # ---------------------------------------------------------
    try:
        logger.info(
            "Loading embedding model: %s",
            model_name
        )
        model = SentenceTransformer(model_name)
        logger.info(
            "Embedding model loaded successfully: %s",
            model_name
        )
    except Exception as error:
        logger.exception(
            "Failed to load embedding model: %s",
            model_name
        )
        return {
            "success": False,
            "model_name": model_name,
            "total_chunks": len(chunks),
            "embedded_chunks": 0,
            "embeddings": [],
            "error": {
                "type": "EMBEDDING_MODEL_ERROR",
                "message": (
                    f"Failed to load embedding model "
                    f"'{model_name}': {error}"
                )
            }
        }

    # ---------------------------------------------------------
    # 3. Extract chunk text
    # ---------------------------------------------------------

    try:
        texts = [
            chunk.page_content
            for chunk in chunks
        ]
    except Exception as error:
        logger.exception(
            "Failed to read text from chunks"
        )
        return {
            "success": False,
            "model_name": model_name,
            "total_chunks": len(chunks),
            "embedded_chunks": 0,
            "embeddings": [],
            "error": {
                "type": "CHUNK_TEXT_ERROR",
                "message": str(error)
            }
        }

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------
    try:
        logger.info(
            "Starting embedding generation for %s chunks",
            len(texts)
        )
        embeddings = model.encode(
            texts,
            show_progress_bar=False
        )
        logger.info(
            "Successfully embedded %s/%s chunks",
            len(embeddings),
            len(chunks)
        )

        # -----------------------------------------------------
        # 5. Return embedding result
        # -----------------------------------------------------

        return {
            "success": True,
            "model_name": model_name,
            "total_chunks": len(chunks),
            "embedded_chunks": len(embeddings),
            "embeddings": embeddings.tolist(),
            "error": None
        }

    except Exception as error:
        logger.exception(
            "Embedding generation failed"
        )

        return {
            "success": False,
            "model_name": model_name,
            "total_chunks": len(chunks),
            "embedded_chunks": 0,
            "embeddings": [],
            "error": {
                "type": "EMBEDDING_GENERATION_ERROR",
                "message": str(error)
            }
        }