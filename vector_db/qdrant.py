import logging
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

logger = logging.getLogger(__name__)

def connect_to_qdrant():
    """
    Connect to the Qdrant Cloud cluster and verify the connection.

    Returns:
        dict: Structured Qdrant connection result.
    """
    # ---------------------------------------------------------
    # 1. Read Qdrant credentials
    # ---------------------------------------------------------
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        logger.error("QDRANT_URL is not configured")

        return {
            "success": False,
            "client": None,
            "error": {
                "type": "QDRANT_URL_MISSING",
                "message": "QDRANT_URL is not configured."
            }
        }

    if not qdrant_api_key:
        logger.error("QDRANT_API_KEY is not configured")
        return {
            "success": False,
            "client": None,
            "error": {
                "type": "QDRANT_API_KEY_MISSING",
                "message": "QDRANT_API_KEY is not configured."
            }
        }

    # ---------------------------------------------------------
    # 2. Create Qdrant client
    # ---------------------------------------------------------
    try:
        logger.info("Connecting to Qdrant Cloud")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # -----------------------------------------------------
        # 3. Verify the connection
        # -----------------------------------------------------
        client.get_collections()
        logger.info("Successfully connected to Qdrant Cloud")
        return {
            "success": True,
            "client": client,
            "error": None
        }

    except Exception as error:
        logger.exception(
            "Failed to connect to Qdrant Cloud"
        )

        return {
            "success": False,
            "client": None,
            "error": {
                "type": "QDRANT_CONNECTION_ERROR",
                "message": str(error)
            }
        }

if __name__ == "__main__":
    response = connect_to_qdrant()
    print(response)