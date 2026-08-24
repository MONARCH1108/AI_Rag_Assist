from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.pdf_text_extraction import extract_pdf_text
from ingestion.text_ingestion import detect_file_type
import logging

def recursive_character_chunking(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split LangChain Documents using RecursiveCharacterTextSplitter.

    Args:
        documents (list): List of LangChain Document objects.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: List of chunked LangChain Document objects.
    """

    logger = logging.getLogger(__name__)
    logger.info("Entering recursive character chunking method")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(
        "Recursive character chunking completed: %s chunks created",
        len(chunks)
    )
    return chunks
