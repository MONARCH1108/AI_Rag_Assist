from ingestion.text_ingestion import detect_file_type
from ingestion.pdf_text_extraction import extract_pdf_text
from chunking.recursive_character_text_splitter import recursive_character_chunking

def main():
    file_path = input("Enter the path to the document: ").strip()

    # ---------------------------------------------------------
    # 1. Detect file type
    # ---------------------------------------------------------
    detection_result = detect_file_type(file_path)
    if not detection_result["success"]:
        print(detection_result)
        return

    # ---------------------------------------------------------
    # 2. Extract PDF text
    # ---------------------------------------------------------
    if detection_result["file_type"] == "pdf":
        extraction_result = extract_pdf_text(file_path)
        if not extraction_result["success"]:
            print(extraction_result)
            return
        documents = extraction_result["documents"]
    else:
        print(
            f"File type '{detection_result['file_type']}' "
            "does not have an extraction pipeline yet."
        )
        return

    # ---------------------------------------------------------
    # 3. Chunk extracted documents
    # ---------------------------------------------------------
    chunks = recursive_character_chunking(documents)

    # ---------------------------------------------------------
    # 4. Basic integration test result
    # ---------------------------------------------------------
    print(f"File type : {detection_result['file_type']}")
    print(f"Pages     : {len(documents)}")
    print(f"Chunks    : {len(chunks)}")


if __name__ == "__main__":
    main()