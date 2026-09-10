import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, Header

from utils.logger import logger
from utils.supabase_service import check_supabase_health
from utils.groq_service import check_groq_health

from ingestion.text_ingestion import detect_file_type
from ingestion.pdf_text_extraction import extract_pdf_text

from chunking.recursive_character_text_splitter import recursive_character_chunking

from embedding.sentence_transformer import embed_chunks

from vector_db.insert_embeddings import insert_documents
from vector_db.supabase_client import connect_to_supabase
from vector_db.list_documents import list_documents


# =============================================================
# ENVIRONMENT CONFIGURATION
# =============================================================

load_dotenv()


# =============================================================
# FASTAPI APPLICATION
# =============================================================

app = FastAPI(
    title="AI RAG Assist",
    description="Retrieval-Augmented Generation backend API",
    version="1.0.0",
)


# =============================================================
# GUEST SESSION VALIDATION
# =============================================================

def validate_guest_session(client, user_id):
    """
    Validate that the supplied user_id belongs to an active
    guest session.

    The user_id is expected to come from the X-User-ID header.
    """

    if not user_id:
        logger.warning(
            "Request rejected because X-User-ID was not provided"
        )

        return {
            "success": False,
            "error": {
                "type": "USER_ID_MISSING",
                "message": "X-User-ID header is required.",
            },
        }

    try:
        result = (
            client
            .table("guest_sessions")
            .select("id, expires_at")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )

        session = result.data

        if not session:
            logger.warning(
                "Guest session not found for user_id: %s",
                user_id,
            )

            return {
                "success": False,
                "error": {
                    "type": "INVALID_USER_ID",
                    "message": "Guest session was not found.",
                },
            }

        # ---------------------------------------------------------
        # Check expiration
        # ---------------------------------------------------------

        expires_at = session.get("expires_at")

        if expires_at:
            expiration_time = datetime.fromisoformat(
                expires_at.replace("Z", "+00:00")
            )

            current_time = datetime.now(timezone.utc)

            if expiration_time <= current_time:
                logger.warning(
                    "Guest session expired for user_id: %s",
                    user_id,
                )

                return {
                    "success": False,
                    "error": {
                        "type": "USER_SESSION_EXPIRED",
                        "message": "Guest session has expired.",
                    },
                }

        # ---------------------------------------------------------
        # Update last activity
        # ---------------------------------------------------------

        client.table("guest_sessions").update(
            {
                "last_activity_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }
        ).eq("id", user_id).execute()

        logger.info(
            "Guest session validated successfully: %s",
            user_id,
        )

        return {
            "success": True,
            "user_id": user_id,
        }

    except Exception as error:
        logger.exception(
            "Failed to validate guest session: %s",
            error,
        )

        return {
            "success": False,
            "error": {
                "type": "GUEST_SESSION_VALIDATION_ERROR",
                "message": str(error),
            },
        }


# =============================================================
# HEALTH API
# =============================================================

@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
)
def health_check():
    logger.info("Starting health check")

    # ---------------------------------------------------------
    # 1. Check Supabase service
    # ---------------------------------------------------------

    supabase_health = check_supabase_health()

    # ---------------------------------------------------------
    # 2. Check Groq service
    # ---------------------------------------------------------

    groq_health = check_groq_health()

    # ---------------------------------------------------------
    # 3. Determine overall application health
    # ---------------------------------------------------------

    if (
        supabase_health["healthy"]
        and groq_health["healthy"]
    ):
        status = "healthy"
    else:
        status = "unhealthy"

    # ---------------------------------------------------------
    # 4. Build health response
    # ---------------------------------------------------------

    response = {
        "status": status,
        "service": "ai-rag-assist",
        "version": app.version,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "dependencies": {
            "supabase": supabase_health,
            "groq": groq_health,
        },
    }

    # ---------------------------------------------------------
    # 5. Log health check result
    # ---------------------------------------------------------

    if status == "healthy":
        logger.info(
            "Health check completed successfully: %s",
            response,
        )
    else:
        logger.error(
            "Health check failed: %s",
            response,
        )

    return response


# =============================================================
# DOCUMENT INGESTION API
# =============================================================

@app.post(
    "/run-pipeline",
    tags=["Pipeline"],
    summary="Upload and process PDF documents",
)
async def upload_documents(
    files: list[UploadFile] = File(...),
    x_user_id: str | None = Header(
        default=None,
        alias="X-User-ID",
    ),
):
    logger.info(
        "Starting document ingestion request: %s file(s)",
        len(files),
    )

    # ---------------------------------------------------------
    # 1. Validate request
    # ---------------------------------------------------------

    if not files:
        logger.warning(
            "Document ingestion request received without files"
        )

        return {
            "success": False,
            "documents": [],
            "error": {
                "type": "NO_FILES_PROVIDED",
                "message": "At least one PDF file is required.",
            },
        }

    # ---------------------------------------------------------
    # 2. Connect to Supabase
    # ---------------------------------------------------------

    supabase_result = connect_to_supabase()

    if not supabase_result["success"]:
        logger.error(
            "Unable to connect to Supabase for document ingestion"
        )

        return {
            "success": False,
            "documents": [],
            "error": supabase_result["error"],
        }

    supabase_client = supabase_result["client"]

    # ---------------------------------------------------------
    # 3. Validate guest session
    # ---------------------------------------------------------

    guest_validation = validate_guest_session(
        client=supabase_client,
        user_id=x_user_id,
    )

    if not guest_validation["success"]:
        return {
            "success": False,
            "documents": [],
            "error": guest_validation["error"],
        }

    user_id = guest_validation["user_id"]

    logger.info(
        "Processing documents for user_id: %s",
        user_id,
    )

    # ---------------------------------------------------------
    # 4. Process uploaded documents
    # ---------------------------------------------------------

    results = []

    for uploaded_file in files:
        file_name = uploaded_file.filename or "unknown"

        logger.info(
            "Starting ingestion for uploaded document: %s "
            "for user_id: %s",
            file_name,
            user_id,
        )

        temporary_path = None

        try:

            # -------------------------------------------------
            # 4.1 Validate filename
            # -------------------------------------------------

            if not uploaded_file.filename:
                logger.warning(
                    "Uploaded file has no filename"
                )

                results.append({
                    "file_name": None,
                    "success": False,
                    "error": {
                        "type": "INVALID_FILE_NAME",
                        "message": (
                            "Uploaded file must have a filename."
                        ),
                    },
                })

                continue

            # -------------------------------------------------
            # 4.2 Create temporary file
            # -------------------------------------------------

            suffix = Path(
                uploaded_file.filename
            ).suffix.lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary_file:

                temporary_path = temporary_file.name

                while True:
                    chunk = await uploaded_file.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    temporary_file.write(chunk)

            logger.info(
                "Temporary file created for %s: %s",
                file_name,
                temporary_path,
            )

            # -------------------------------------------------
            # 4.3 Detect file type
            # -------------------------------------------------

            detection_result = detect_file_type(
                temporary_path
            )

            if not detection_result["success"]:
                logger.warning(
                    "File type detection failed: %s",
                    file_name,
                )

                results.append({
                    "file_name": file_name,
                    "success": False,
                    "error": detection_result["error"],
                })

                continue

            # -------------------------------------------------
            # 4.4 Only PDF is currently supported
            # -------------------------------------------------

            if detection_result["file_type"] != "pdf":
                logger.warning(
                    "Unsupported document type received: %s",
                    detection_result["file_type"],
                )

                results.append({
                    "file_name": file_name,
                    "success": False,
                    "file_type": (
                        detection_result["file_type"]
                    ),
                    "error": {
                        "type": "UNSUPPORTED_FILE_TYPE",
                        "message": (
                            "Only PDF documents are currently "
                            "supported."
                        ),
                    },
                })

                continue

            # -------------------------------------------------
            # 4.5 Extract PDF text
            # -------------------------------------------------

            extraction_result = extract_pdf_text(
                temporary_path,
                user_id=user_id,
            )

            if not extraction_result["success"]:
                logger.error(
                    "PDF extraction failed: %s",
                    file_name,
                )

                results.append({
                    "file_name": file_name,
                    "success": False,
                    "file_type": "pdf",
                    "error": extraction_result["error"],
                })

                continue

            extracted_documents = (
                extraction_result["documents"]
            )

            # -------------------------------------------------
            # Restore original uploaded file metadata
            # -------------------------------------------------

            for document in extracted_documents:
                document.metadata["file_name"] = file_name
                document.metadata["source"] = file_name
                document.metadata["user_id"] = user_id

            # -------------------------------------------------
            # 4.6 Chunk document
            # -------------------------------------------------

            chunks = recursive_character_chunking(
                extracted_documents
            )

            if not chunks:
                logger.warning(
                    "No chunks generated for document: %s",
                    file_name,
                )

                results.append({
                    "file_name": file_name,
                    "success": False,
                    "file_type": "pdf",
                    "error": {
                        "type": "NO_CHUNKS_GENERATED",
                        "message": (
                            "No chunks could be generated "
                            "from the PDF."
                        ),
                    },
                })

                continue

            # -------------------------------------------------
            # 4.7 Generate embeddings
            # -------------------------------------------------

            embedding_result = embed_chunks(chunks)

            if not embedding_result["success"]:
                logger.error(
                    "Embedding generation failed: %s",
                    file_name,
                )

                results.append({
                    "file_name": file_name,
                    "success": False,
                    "file_type": "pdf",
                    "error": embedding_result["error"],
                })

                continue

            # -------------------------------------------------
            # 4.8 Insert vectors into Supabase
            # -------------------------------------------------

            insertion_result = insert_documents(
                client=supabase_client,
                chunks=chunks,
                embeddings=embedding_result["embeddings"],
                user_id=user_id,
            )

            if not insertion_result["success"]:
                logger.error(
                    "Supabase insertion failed: %s",
                    file_name,
                )

                results.append({
                    "file_name": file_name,
                    "success": False,
                    "file_type": "pdf",
                    "error": insertion_result["error"],
                })

                continue

            # -------------------------------------------------
            # 4.9 Successful ingestion
            # -------------------------------------------------

            logger.info(
                "Document ingestion completed successfully: "
                "%s for user_id: %s",
                file_name,
                user_id,
            )

            results.append({
                "file_name": file_name,
                "success": True,
                "file_type": "pdf",
                "pages": len(extracted_documents),
                "chunks": len(chunks),
                "embedded_chunks": (
                    embedding_result["embedded_chunks"]
                ),
                "inserted_vectors": (
                    insertion_result["inserted_count"]
                ),
                "embedding_model": (
                    embedding_result["model_name"]
                ),
                "error": None,
            })

        except Exception as error:
            logger.exception(
                "Unexpected document ingestion error: %s",
                file_name,
            )

            results.append({
                "file_name": file_name,
                "success": False,
                "error": {
                    "type": "DOCUMENT_INGESTION_ERROR",
                    "message": str(error),
                },
            })

        finally:

            # -------------------------------------------------
            # 4.10 Remove temporary file
            # -------------------------------------------------

            if (
                temporary_path
                and os.path.exists(temporary_path)
            ):
                try:
                    os.remove(temporary_path)

                    logger.info(
                        "Temporary file removed: %s",
                        temporary_path,
                    )

                except Exception:
                    logger.exception(
                        "Failed to remove temporary file: %s",
                        temporary_path,
                    )

            await uploaded_file.close()

    # ---------------------------------------------------------
    # 5. Determine overall result
    # ---------------------------------------------------------

    successful_documents = [
        result
        for result in results
        if result["success"]
    ]

    failed_documents = [
        result
        for result in results
        if not result["success"]
    ]

    overall_success = (
        len(successful_documents) > 0
        and len(failed_documents) == 0
    )

    # ---------------------------------------------------------
    # 6. Return response
    # ---------------------------------------------------------

    response = {
        "success": overall_success,
        "user_id": user_id,
        "total_files": len(results),
        "successful_files": len(
            successful_documents
        ),
        "failed_files": len(
            failed_documents
        ),
        "documents": results,
    }

    if overall_success:
        logger.info(
            "Document ingestion request completed successfully "
            "for user_id: %s",
            user_id,
        )
    else:
        logger.warning(
            "Document ingestion request completed with failures "
            "for user_id: %s",
            user_id,
        )

    return response


# =============================================================
# DOCUMENT LIST API
# =============================================================

@app.get(
    "/documents",
    tags=["Documents"],
    summary="List stored documents",
)
async def get_documents(
    x_user_id: str | None = Header(
        default=None,
        alias="X-User-ID",
    ),
):
    """
    Return documents belonging only to the supplied guest user.
    """

    logger.info(
        "Starting document listing request"
    )

    # ---------------------------------------------------------
    # 1. Connect to Supabase
    # ---------------------------------------------------------

    supabase_connection = connect_to_supabase()

    if not supabase_connection["success"]:
        logger.error(
            "Unable to connect to Supabase for document listing"
        )

        return {
            "success": False,
            "documents": [],
            "error": supabase_connection["error"],
        }

    supabase_client = supabase_connection["client"]

    # ---------------------------------------------------------
    # 2. Validate guest session
    # ---------------------------------------------------------

    guest_validation = validate_guest_session(
        client=supabase_client,
        user_id=x_user_id,
    )

    if not guest_validation["success"]:
        return {
            "success": False,
            "documents": [],
            "error": guest_validation["error"],
        }

    user_id = guest_validation["user_id"]

    # ---------------------------------------------------------
    # 3. List documents for this user only
    # ---------------------------------------------------------

    result = list_documents(
        client=supabase_client,
        user_id=user_id,
    )

    return result