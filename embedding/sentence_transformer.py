import logging
import os

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

# =============================================================
# EMBEDDING CONFIGURATION
# =============================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cached model instances
_MODEL_CACHE = {}


def _get_embedding_model(model_name):
    """
    Load and cache the Sentence Transformer model.

    The model is loaded only once per model name and reused
    for subsequent embedding operations.
    """

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        logger.info(
            "Loading embedding model: %s",
            model_name
        )

        hf_token = os.getenv("HF_TOKEN")

        if hf_token:
            model = SentenceTransformer(
                model_name,
                token=hf_token
            )
        else:
            model = SentenceTransformer(model_name)

        _MODEL_CACHE[model_name] = model

        logger.info(
            "Embedding model ready: %s",
            model_name
        )

        return model

    except Exception as error:
        logger.exception(
            "Failed to load embedding model"
        )
        raise error


def embed_chunks(
    chunks,
    model_name=EMBEDDING_MODEL_NAME
):
    """
    Generate vector embeddings for LangChain document chunks.

    Args:
        chunks (list): List of LangChain Document objects.
        model_name (str): Hugging Face Sentence Transformer
            model name.

    Returns:
        dict: Structured embedding result.
    """

    # ---------------------------------------------------------
    # 1. Validate chunks
    # ---------------------------------------------------------

    if not chunks:
        logger.warning(
            "No chunks provided for encoding"
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
    # 2. Load / retrieve cached model
    # ---------------------------------------------------------

    try:
        model = _get_embedding_model(model_name)

    except Exception as error:
        return {
            "success": False,
            "model_name": model_name,
            "total_chunks": len(chunks),
            "embedded_chunks": 0,
            "embeddings": [],
            "error": {
                "type": "EMBEDDING_MODEL_ERROR",
                "message": str(error)
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
            "Failed to extract chunk text"
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
    # 4. Encode chunks
    # ---------------------------------------------------------

    try:
        logger.info(
            "Encoding %s chunks using %s",
            len(texts),
            model_name
        )

        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        logger.info(
            "Encoding completed: %s/%s chunks",
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
            "Chunk encoding failed"
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