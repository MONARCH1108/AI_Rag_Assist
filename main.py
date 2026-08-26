from dotenv import load_dotenv

load_dotenv()

from ingestion.text_ingestion import detect_file_type
from ingestion.pdf_text_extraction import extract_pdf_text
from chunking.recursive_character_text_splitter import (
    recursive_character_chunking
)
from embedding.sentence_transformer import embed_chunks
from vector_db.supabase_client import connect_to_supabase
from vector_db.verify_database import verify_supabase_database
from vector_db.insert_embeddings import insert_documents
from vector_db.list_documents import list_documents
from vector_db.query_vectors import query_vectors
from llm.groq import generate_answer

def query_existing_document(
    supabase_client,
    documents
):
    """
    Query documents that are already stored in Supabase
    and generate an answer using the retrieved context.
    """
    print("\nExisting documents:\n")
    for index, document in enumerate(
        documents,
        start=1
    ):
        print(
            f"{index}. "
            f"{document['file_name']}"
        )
    print()
    # ---------------------------------------------------------
    # 1. Select document
    # ---------------------------------------------------------

    while True:
        try:
            selection = int(
                input(
                    "Select a document number: "
                ).strip()
            )
            if 1 <= selection <= len(documents):
                break
            print(
                "Invalid document number."
            )
        except ValueError:
            print(
                "Please enter a valid number."
            )
    selected_document = documents[
        selection - 1
    ]
    print(
        f"\nSelected document: "
        f"{selected_document['file_name']}"
    )

    # ---------------------------------------------------------
    # 2. Get user query
    # ---------------------------------------------------------

    query = input(
        "\nEnter your query: "
    ).strip()

    # ---------------------------------------------------------
    # 3. Retrieve relevant chunks from Supabase
    # ---------------------------------------------------------

    query_result = query_vectors(
        client=supabase_client,
        query=query,
        top_k=5,
    )
    if not query_result["success"]:
        print(query_result)
        return

    # ---------------------------------------------------------
    # 4. Generate answer using retrieved context
    # ---------------------------------------------------------

    llm_result = generate_answer(
        question=query,
        retrieved_chunks=query_result["results"],
    )
    if not llm_result["success"]:
        print(llm_result)
        return
    # ---------------------------------------------------------
    # 5. Display final answer
    # ---------------------------------------------------------

    print("\nAnswer:\n")
    print(
        llm_result["answer"]
    )

def process_new_document(
    supabase_client
):
    """
    Run the complete ingestion pipeline for a new document
    and then query the newly added document.
    """
    file_path = input(
        "\nEnter the path to the document: "
    ).strip()

    # ---------------------------------------------------------
    # 1. Detect file type
    # ---------------------------------------------------------

    detection_result = detect_file_type(
        file_path
    )
    if not detection_result["success"]:
        print(detection_result)
        return

    # ---------------------------------------------------------
    # 2. Extract document
    # ---------------------------------------------------------
    if detection_result["file_type"] == "pdf":
        extraction_result = extract_pdf_text(
            file_path
        )
        if not extraction_result["success"]:
            print(extraction_result)
            return
        documents = extraction_result[
            "documents"
        ]
    else:
        print(
            f"File type "
            f"'{detection_result['file_type']}' "
            "does not have an extraction pipeline yet."
        )
        return

    # ---------------------------------------------------------
    # 3. Chunk document
    # ---------------------------------------------------------

    chunks = recursive_character_chunking(
        documents
    )

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------

    embedding_result = embed_chunks(
        chunks
    )
    if not embedding_result["success"]:
        print(embedding_result)
        return

    # ---------------------------------------------------------
    # 5. Insert embeddings into Supabase
    # ---------------------------------------------------------

    insertion_result = insert_documents(
        client=supabase_client,
        chunks=chunks,
        embeddings=embedding_result["embeddings"],
    )
    if not insertion_result["success"]:
        print(insertion_result)
        return

    # ---------------------------------------------------------
    # 6. Pipeline result
    # ---------------------------------------------------------

    print(
        "\nDocument successfully added.\n"
    )
    print(
        f"File type       : "
        f"{detection_result['file_type']}"
    )
    print(
        f"Pages           : "
        f"{len(documents)}"
    )
    print(
        f"Chunks          : "
        f"{len(chunks)}"
    )
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

    # ---------------------------------------------------------
    # 7. Query the newly added document
    # ---------------------------------------------------------

    query = input(
        "\nEnter your query: "
    ).strip()

    # ---------------------------------------------------------
    # 8. Retrieve relevant chunks
    # ---------------------------------------------------------

    query_result = query_vectors(
        client=supabase_client,
        query=query,
        top_k=5,
    )

    if not query_result["success"]:
        print(query_result)
        return

    # ---------------------------------------------------------
    # 9. Generate answer using retrieved context
    # ---------------------------------------------------------

    llm_result = generate_answer(
        question=query,
        retrieved_chunks=query_result["results"],
    )
    if not llm_result["success"]:
        print(llm_result)
        return

    # ---------------------------------------------------------
    # 10. Display final answer
    # ---------------------------------------------------------

    print("\nAnswer:\n")
    print(
        llm_result["answer"]
    )


def main():

    # =========================================================
    # 1. Connect to Supabase
    # =========================================================

    supabase_result = connect_to_supabase()
    if not supabase_result["success"]:
        print(supabase_result)
        return
    supabase_client = supabase_result[
        "client"
    ]

    # =========================================================
    # 2. Verify Supabase database
    # =========================================================

    database_result = verify_supabase_database(
        supabase_client
    )

    if not database_result["success"]:
        print(database_result)
        return

    # =========================================================
    # 3. Get existing documents
    # =========================================================

    documents_result = list_documents(
        client=supabase_client
    )
    if not documents_result["success"]:
        print(documents_result)
        return
    existing_documents = documents_result[
        "documents"
    ]

    # =========================================================
    # 4. Choose existing or new document
    # =========================================================

    print(
        "\nWhat would you like to do?"
    )
    print(
        "1. Query an existing document"
    )
    print(
        "2. Add a new document"
    )
    choice = input(
        "\nEnter your choice: "
    ).strip()

    # =========================================================
    # 5. Existing document
    # =========================================================

    if choice == "1":
        if not existing_documents:
            print(
                "\nNo documents are currently "
                "stored in the vector database."
            )
            return
        query_existing_document(
            supabase_client,
            existing_documents
        )

    # =========================================================
    # 6. New document
    # =========================================================
    elif choice == "2":
        process_new_document(
            supabase_client
        )
    else:
        print(
            "\nInvalid choice."
        )

if __name__ == "__main__":
    main()