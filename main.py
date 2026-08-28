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

from delete.delete_files import (
    delete_document,
    delete_all_documents
)

from llm.groq import generate_answer


def query_existing_document(
    supabase_client,
    documents
):
    """
    Query an existing document and generate an answer
    using the retrieved RAG context.
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

    while True:
        try:
            selection = int(
                input(
                    "Select a document number: "
                ).strip()
            )

            if 1 <= selection <= len(documents):
                break

            print("Invalid document number.")

        except ValueError:
            print("Please enter a valid number.")

    selected_document = documents[
        selection - 1
    ]

    print(
        f"\nSelected document: "
        f"{selected_document['file_name']}"
    )

    query = input(
        "\nEnter your query: "
    ).strip()

    query_result = query_vectors(
        client=supabase_client,
        query=query,
        top_k=5,
    )

    if not query_result["success"]:
        print(query_result)
        return

    llm_result = generate_answer(
        question=query,
        retrieved_chunks=query_result["results"],
    )

    if not llm_result["success"]:
        print(llm_result)
        return

    print("\nAnswer:\n")
    print(llm_result["answer"])


def process_new_document(
    supabase_client
):
    """
    Run the complete document ingestion pipeline.
    """

    file_path = input(
        "\nEnter the path to the document: "
    ).strip()

    detection_result = detect_file_type(
        file_path
    )

    if not detection_result["success"]:
        print(detection_result)
        return

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

    chunks = recursive_character_chunking(
        documents
    )

    embedding_result = embed_chunks(
        chunks
    )

    if not embedding_result["success"]:
        print(embedding_result)
        return

    insertion_result = insert_documents(
        client=supabase_client,
        chunks=chunks,
        embeddings=embedding_result["embeddings"],
    )

    if not insertion_result["success"]:
        print(insertion_result)
        return

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

    query = input(
        "\nEnter your query: "
    ).strip()

    query_result = query_vectors(
        client=supabase_client,
        query=query,
        top_k=5,
    )

    if not query_result["success"]:
        print(query_result)
        return

    llm_result = generate_answer(
        question=query,
        retrieved_chunks=query_result["results"],
    )

    if not llm_result["success"]:
        print(llm_result)
        return

    print("\nAnswer:\n")
    print(llm_result["answer"])


def delete_one_document(
    supabase_client,
    documents
):
    """
    Allow the user to select one document and delete
    all chunks belonging to that document.
    """

    if not documents:
        print(
            "\nNo documents are currently stored."
        )
        return

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

    while True:
        try:
            selection = int(
                input(
                    "Select a document to delete: "
                ).strip()
            )

            if 1 <= selection <= len(documents):
                break

            print("Invalid document number.")

        except ValueError:
            print("Please enter a valid number.")

    selected_document = documents[
        selection - 1
    ]

    file_name = selected_document[
        "file_name"
    ]

    confirmation = input(
        f"\nDelete '{file_name}'? (y/n): "
    ).strip().lower()

    if confirmation != "y":
        print("\nDeletion cancelled.")
        return

    deletion_result = delete_document(
        client=supabase_client,
        file_name=file_name,
    )

    if not deletion_result["success"]:
        print(deletion_result)
        return

    print(
        f"\nSuccessfully deleted "
        f"'{file_name}'."
    )

    print(
        f"Deleted chunks: "
        f"{deletion_result['deleted_count']}"
    )


def delete_all(
    supabase_client,
    documents
):
    """
    Delete all documents and their chunks from Supabase.
    """

    if not documents:
        print(
            "\nNo documents are currently stored."
        )
        return

    print(
        f"\nWARNING: This will delete ALL "
        f"{len(documents)} documents."
    )

    confirmation = input(
        "\nType 'DELETE' to confirm: "
    ).strip()

    if confirmation != "DELETE":
        print("\nDeletion cancelled.")
        return

    deletion_result = delete_all_documents(
        client=supabase_client
    )

    if not deletion_result["success"]:
        print(deletion_result)
        return

    print(
        "\nAll documents successfully deleted."
    )

    print(
        f"Deleted chunks: "
        f"{deletion_result['deleted_count']}"
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
    # 2. Verify database
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
    # 4. Main menu
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

    print(
        "3. Delete a document"
    )

    print(
        "4. Delete all documents"
    )

    choice = input(
        "\nEnter your choice: "
    ).strip()

    # =========================================================
    # 5. Query existing document
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
    # 6. Add new document
    # =========================================================

    elif choice == "2":

        process_new_document(
            supabase_client
        )

    # =========================================================
    # 7. Delete one document
    # =========================================================

    elif choice == "3":

        delete_one_document(
            supabase_client,
            existing_documents
        )

    # =========================================================
    # 8. Delete all documents
    # =========================================================

    elif choice == "4":

        delete_all(
            supabase_client,
            existing_documents
        )

    # =========================================================
    # 9. Invalid choice
    # =========================================================

    else:

        print(
            "\nInvalid choice."
        )


if __name__ == "__main__":
    main()