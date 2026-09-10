import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile

from utils.logger import logger
from utils.supabase_service import check_supabase_health
from utils.groq_service import check_groq_health

from ingestion.text_ingestion import detect_file_type
from ingestion.pdf_text_extraction import extract_pdf_text

from chunking.recursive_character_text_splitter import recursive_character_chunking

from embedding.sentence_transformer import embed_chunks

from vector_db.insert_embeddings import insert_documents
from vector_db.supabase_client import connect_to_supabase


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
# HEALTH API
# =============================================================

@app.get("/health", tags=["Health"], summary="Health check",)
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

@app.post("/documents", tags=["Documents"], summary="Upload and process PDF documents")
async def upload_documents(files: list[UploadFile] = File(...)):
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
    # 3. Process uploaded documents
    # ---------------------------------------------------------

    results = []
    for uploaded_file in files:
        file_name = uploaded_file.filename or "unknown"
        logger.info(
            "Starting ingestion for uploaded document: %s",
            file_name,
        )
        temporary_path = None
        try:
            # -------------------------------------------------
            # 3.1 Validate filename
            # -------------------------------------------------

            if not uploaded_file.filename:
                logger.warning("Uploaded file has no filename")
                results.append({
                    "file_name": None,
                    "success": False,
                    "error": {
                        "type": "INVALID_FILE_NAME",
                        "message": "Uploaded file must have a filename.",
                    },
                })
                continue

            # -------------------------------------------------
            # 3.2 Create temporary file
            # -------------------------------------------------

            suffix = Path(uploaded_file.filename).suffix.lower()
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
            # 3.3 Detect file type
            # -------------------------------------------------

            detection_result = detect_file_type(temporary_path)
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
            # 3.4 Only PDF is currently supported
            # -------------------------------------------------

            if detection_result["file_type"] != "pdf":
                logger.warning(
                    "Unsupported document type received: %s",
                    detection_result["file_type"],
                )
                results.append({
                    "file_name": file_name,
                    "success": False,
                    "file_type": detection_result["file_type"],
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
            # 3.5 Extract PDF text
            # -------------------------------------------------

            extraction_result = extract_pdf_text(
                temporary_path
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
            # 3.6 Chunk document
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
            # 3.7 Generate embeddings
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
            # 3.8 Insert vectors into Supabase
            # -------------------------------------------------

            insertion_result = insert_documents(
                client=supabase_client,
                chunks=chunks,
                embeddings=embedding_result["embeddings"],
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
            # 3.9 Successful ingestion
            # -------------------------------------------------

            logger.info(
                "Document ingestion completed successfully: %s",
                file_name,
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
            # 3.10 Remove temporary file
            # -------------------------------------------------

            if temporary_path and os.path.exists(temporary_path):
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
    # 4. Determine overall result
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
    # 5. Return response
    # ---------------------------------------------------------

    response = {
        "success": overall_success,
        "total_files": len(results),
        "successful_files": len(successful_documents),
        "failed_files": len(failed_documents),
        "documents": results,
    }

    if overall_success:
        logger.info(
            "Document ingestion request completed successfully"
        )
    else:
        logger.warning(
            "Document ingestion request completed with failures"
        )

    return response