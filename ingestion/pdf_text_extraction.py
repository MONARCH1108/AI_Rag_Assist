from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from ingestion.text_ingestion import detect_file_type
from utils.logger import logger

def extract_pdf_text(file_path, user_id=None):
    """
    Extract text from a PDF.

    Args:
        file_path (str): Path to the PDF file.
        user_id (str): Unique identifier of the user/guest uploading the PDF.

    Returns:
        dict: PDF extraction result containing LangChain Documents.
    """
    detection_result = detect_file_type(file_path)
    if not detection_result["success"]:
        return detection_result
    if detection_result["file_type"] != "pdf":
        return {
            "success": False,
            "file_name": detection_result["file_name"],
            "file_type": detection_result["file_type"],
            "documents": [],
            "error": {
                "type": "INVALID_EXTRACTION_TYPE",
                "message": (
                    f"PDF extraction was requested, but the detected "
                    f"file type is '{detection_result['file_type']}'."
                )
            }
        }
    try:
        logger.info(
            "Starting PDF text extraction: %s",
            detection_result["file_name"]
        )
        reader = PdfReader(file_path)
        logger.info(
            "PDF contains %s pages",
            len(reader.pages)
        )
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            logger.info(
                "Extracting text from page %s",
                page_number
            )
            text = page.extract_text() or ""
            document = Document(
                page_content=text,
                metadata={
                    "source": str(Path(file_path)),
                    "file_name": detection_result["file_name"],
                    "file_type": "pdf",
                    "page": page_number,
                    "user_id": user_id,
                }
            )
            documents.append(document)
        logger.info(
            "PDF text extraction completed: %s",
            detection_result["file_name"]
        )
        return {
            "success": True,
            "file_name": detection_result["file_name"],
            "file_type": "pdf",
            "documents": documents,
            "error": None
        }

    except Exception as error:
        logger.exception(
            "PDF text extraction failed: %s",
            detection_result["file_name"]
        )
        return {
            "success": False,
            "file_name": detection_result["file_name"],
            "file_type": "pdf",
            "documents": [],
            "error": {
                "type": "PDF_EXTRACTION_ERROR",
                "message": str(error)
            }
        }

if __name__ == "__main__":
    file_path = input("Enter the path to the PDF: ").strip()
    response = extract_pdf_text(file_path)
    print(response)