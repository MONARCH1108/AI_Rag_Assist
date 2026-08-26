from ingestion.text_ingestion import detect_file_type
from ingestion.pdf_text_extraction import extract_pdf_text
from chunking.recursive_character_text_splitter import recursive_character_chunking
from embedding.sentence_transformer import embed_chunks
from vector_db.qdrant import connect_to_qdrant
from vector_db.create_collection import create_qdrant_collection
from vector_db.insert_embeddings import insert_embeddings

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
    # 4. Generate embeddings
    # ---------------------------------------------------------
    embedding_result = embed_chunks(chunks)
    if not embedding_result["success"]:
        print(embedding_result)
        return

    # ---------------------------------------------------------
    # 5. Connect to Qdrant
    # ---------------------------------------------------------
    qdrant_result = connect_to_qdrant()

    if not qdrant_result["success"]:
        print(qdrant_result)
        return

    qdrant_client = qdrant_result["client"]

    # ---------------------------------------------------------
    # 6. Create Qdrant collection if required
    # ---------------------------------------------------------
    collection_result = create_qdrant_collection(
        qdrant_client
    )

    if not collection_result["success"]:
        print(collection_result)
        return

    # ---------------------------------------------------------
    # 7. Insert embeddings into Qdrant
    # ---------------------------------------------------------
    insertion_result = insert_embeddings(
        client=qdrant_client,
        chunks=chunks,
        embeddings=embedding_result["embeddings"],
    )

    if not insertion_result["success"]:
        print(insertion_result)
        return

    # ---------------------------------------------------------
    # 8. Basic integration test result
    # ---------------------------------------------------------
    print(f"File type       : {detection_result['file_type']}")
    print(f"Pages           : {len(documents)}")
    print(f"Chunks          : {len(chunks)}")
    print(
        f"Embedded chunks : "
        f"{embedding_result['embedded_chunks']}"
    )
    print(
        f"Embedding model : "
        f"{embedding_result['model_name']}"
    )
    print(
        f"Inserted vectors: "
        f"{insertion_result['inserted_count']}"
    )


if __name__ == "__main__":
    main()