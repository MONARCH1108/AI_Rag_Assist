import os

from groq import Groq
from utils.logger import logger


# =============================================================
# GROQ SERVICE
# =============================================================

def check_groq_health():
    """
    Check the complete Groq LLM service health.

    Verifies:
        1. GROQ_API_KEY is configured.
        2. MODEL_NAME is configured.
        3. Groq client can be initialized.
        4. Groq service can be reached.
        5. The configured LLM model is available and accessible.

    Returns:
        dict:
            Structured Groq health result.
    """

    logger.info(
        "Checking Groq LLM service health"
    )

    # ---------------------------------------------------------
    # 1. Read Groq configuration
    # ---------------------------------------------------------

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model_name = os.getenv("MODEL_NAME")

    # ---------------------------------------------------------
    # 2. Validate Groq API key
    # ---------------------------------------------------------

    if not groq_api_key:
        logger.error(
            "GROQ_API_KEY is not configured"
        )

        return {
            "healthy": False,
            "service": "groq",
            "connection": False,
            "llm_model": groq_model_name,
            "model_access": False,
            "error": {
                "type": "GROQ_CONFIGURATION_ERROR",
                "message": "GROQ_API_KEY is not configured.",
            },
        }

    # ---------------------------------------------------------
    # 3. Validate LLM model configuration
    # ---------------------------------------------------------

    if not groq_model_name:
        logger.error(
            "MODEL_NAME is not configured"
        )

        return {
            "healthy": False,
            "service": "groq",
            "connection": False,
            "llm_model": None,
            "model_access": False,
            "error": {
                "type": "GROQ_MODEL_CONFIGURATION_ERROR",
                "message": "MODEL_NAME is not configured.",
            },
        }

    # ---------------------------------------------------------
    # 4. Initialize Groq client
    # ---------------------------------------------------------

    try:
        logger.info(
            "Initializing Groq client"
        )

        client = Groq(
            api_key=groq_api_key,
        )

        logger.info(
            "Groq client initialized successfully"
        )

    except Exception as error:
        logger.exception(
            "Failed to initialize Groq client"
        )

        return {
            "healthy": False,
            "service": "groq",
            "connection": False,
            "llm_model": groq_model_name,
            "model_access": False,
            "error": {
                "type": "GROQ_CLIENT_ERROR",
                "message": str(error),
            },
        }

    # ---------------------------------------------------------
    # 5. Check Groq service connection and model access
    # ---------------------------------------------------------

    try:
        logger.info(
            "Checking Groq service connection using model: %s",
            groq_model_name,
        )

        response = client.chat.completions.create(
            model=groq_model_name,
            messages=[
                {
                    "role": "user",
                    "content": "hi",
                }
            ],
            temperature=0,
            max_tokens=5,
        )

        # -----------------------------------------------------
        # 6. Validate response
        # -----------------------------------------------------

        if not response:
            logger.error(
                "Groq returned an empty response"
            )

            return {
                "healthy": False,
                "service": "groq",
                "connection": False,
                "llm_model": groq_model_name,
                "model_access": False,
                "error": {
                    "type": "GROQ_EMPTY_RESPONSE",
                    "message": "Groq returned an empty response.",
                },
            }

        logger.info(
            "Groq service connection is healthy"
        )

        logger.info(
            "Configured Groq LLM model is accessible: %s",
            groq_model_name,
        )

        return {
            "healthy": True,
            "service": "groq",
            "connection": True,
            "llm_model": groq_model_name,
            "model_access": True,
            "error": None,
        }

    except Exception as error:
        logger.exception(
            "Groq service or LLM model health check failed"
        )

        return {
            "healthy": False,
            "service": "groq",
            "connection": False,
            "llm_model": groq_model_name,
            "model_access": False,
            "error": {
                "type": "GROQ_CONNECTION_ERROR",
                "message": str(error),
            },
        }