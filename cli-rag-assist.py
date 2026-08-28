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


# =============================================================
# DOCUMENT LIST
# =============================================================

def get_existing_documents(
    supabase_client
):
    """
    Retrieve the current list of unique documents
    stored in the Supabase vector database.

    Returns:
        list:
            List of document objects.
    """

    documents_result = list_documents(
        client=supabase_client
    )

    if not documents_result["success"]:

        print(
            "\nFailed to retrieve documents."
        )

        print(
            documents_result
        )

        return []

    return documents_result[
        "documents"
    ]


# =============================================================
# DOCUMENT SELECTION
# =============================================================

def select_documents(
    documents
):
    """
    Allow the user to select one or more documents.

    Supported input:

        1
            Select document 1.

        1,3
            Select documents 1 and 3.

        all
            Select all documents.

    Returns:
        list:
            Selected document file names.

        None:
            If selection is cancelled.
    """

    if not documents:

        print(
            "\nNo documents are currently "
            "stored in the vector database."
        )

        return None

    print(
        "\nExisting documents:\n"
    )

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

        selection = input(
            "Select document(s) "
            "(example: 1 or 1,3 or all, "
            "or 'back'): "
        ).strip()

        if not selection:
            print(
                "Please select at least one document."
            )
            continue

        # -----------------------------------------------------
        # Back
        # -----------------------------------------------------

        if selection.lower() == "back":

            return None

        # -----------------------------------------------------
        # Select all
        # -----------------------------------------------------

        if selection.lower() == "all":

            selected_documents = [
                document["file_name"]
                for document in documents
            ]

            break

        # -----------------------------------------------------
        # Select one or multiple documents
        # -----------------------------------------------------

        try:

            selected_indexes = [
                int(value.strip())
                for value in selection.split(",")
            ]

        except ValueError:

            print(
                "Please enter valid document numbers."
            )

            continue

        # -----------------------------------------------------
        # Remove duplicate indexes
        # -----------------------------------------------------

        selected_indexes = list(
            dict.fromkeys(
                selected_indexes
            )
        )

        # -----------------------------------------------------
        # Validate indexes
        # -----------------------------------------------------

        if not all(
            1 <= index <= len(documents)
            for index in selected_indexes
        ):

            print(
                "Invalid document number."
            )

            continue

        # -----------------------------------------------------
        # Create selected document list
        # -----------------------------------------------------

        selected_documents = [
            documents[index - 1]["file_name"]
            for index in selected_indexes
        ]

        break

    # ---------------------------------------------------------
    # Display selection
    # ---------------------------------------------------------

    print(
        "\nSelected document(s):"
    )

    for document in selected_documents:

        print(
            f"- {document}"
        )

    return selected_documents


# =============================================================
# GENERATE ANSWER
# =============================================================

def execute_query(
    supabase_client,
    selected_documents,
    query
):
    """
    Execute the complete RAG query pipeline:

        Query
        ↓
        Embedding
        ↓
        Supabase vector search
        ↓
        Retrieved chunks
        ↓
        Groq LLM
        ↓
        Answer
    """

    if not query or not query.strip():

        print(
            "\nQuery cannot be empty."
        )

        return False

    # ---------------------------------------------------------
    # 1. Retrieve relevant chunks
    # ---------------------------------------------------------

    query_result = query_vectors(
        client=supabase_client,
        query=query,
        documents=selected_documents,
        top_k=5,
    )

    if not query_result["success"]:

        print(
            "\nVector search failed:"
        )

        print(
            query_result
        )

        return False

    # ---------------------------------------------------------
    # 2. Check retrieval results
    # ---------------------------------------------------------

    if not query_result["results"]:

        print(
            "\nNo relevant chunks were found "
            "in the selected document(s)."
        )

        return True

    # ---------------------------------------------------------
    # 3. Generate answer
    # ---------------------------------------------------------

    llm_result = generate_answer(
        question=query,
        retrieved_chunks=query_result["results"],
    )

    if not llm_result["success"]:

        print(
            "\nLLM generation failed:"
        )

        print(
            llm_result
        )

        return False

    # ---------------------------------------------------------
    # 4. Display answer
    # ---------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        "ANSWER"
    )

    print(
        "============================================================"
    )

    print(
        llm_result["answer"]
    )

    print(
        "============================================================"
    )

    return True


# =============================================================
# QUERY SESSION
# =============================================================

def query_documents_session(
    supabase_client,
    documents
):
    """
    Start a persistent query session.

    The user selects one or more documents once and can then
    continue asking questions against the same selected
    documents.

    After every query the user can:

        1. Ask another query
        2. Change document selection
        3. Add a new document
        4. Delete a document
        5. Return to main menu
        6. Exit
    """

    # ---------------------------------------------------------
    # Initial document selection
    # ---------------------------------------------------------

    selected_documents = select_documents(
        documents
    )

    if selected_documents is None:

        return "menu"

    # ---------------------------------------------------------
    # Persistent query loop
    # ---------------------------------------------------------

    while True:

        print(
            "\n------------------------------------------------------------"
        )

        print(
            "Currently selected document(s):"
        )

        for document in selected_documents:

            print(
                f"- {document}"
            )

        print(
            "------------------------------------------------------------"
        )

        query = input(
            "\nEnter your query "
            "(or type 'back' to return to menu): "
        ).strip()

        # -----------------------------------------------------
        # Return to menu
        # -----------------------------------------------------

        if query.lower() == "back":

            return "menu"

        if not query:

            print(
                "\nQuery cannot be empty."
            )

            continue

        # -----------------------------------------------------
        # Execute query
        # -----------------------------------------------------

        execute_query(
            supabase_client=supabase_client,
            selected_documents=selected_documents,
            query=query
        )

        # -----------------------------------------------------
        # Next action
        # -----------------------------------------------------

        while True:

            print(
                "\nWhat would you like to do next?"
            )

            print(
                "1. Ask another query"
            )

            print(
                "2. Change document selection"
            )

            print(
                "3. Add a new document"
            )

            print(
                "4. Delete a document"
            )

            print(
                "5. Return to main menu"
            )

            print(
                "6. Exit"
            )

            next_action = input(
                "\nEnter your choice: "
            ).strip()

            # -------------------------------------------------
            # Another query
            # -------------------------------------------------

            if next_action == "1":

                break

            # -------------------------------------------------
            # Change document selection
            # -------------------------------------------------

            elif next_action == "2":

                current_documents = get_existing_documents(
                    supabase_client
                )

                if not current_documents:

                    print(
                        "\nNo documents are available."
                    )

                    return "menu"

                new_selection = select_documents(
                    current_documents
                )

                if new_selection is not None:

                    selected_documents = new_selection

                break

            # -------------------------------------------------
            # Add document
            # -------------------------------------------------

            elif next_action == "3":

                process_new_document(
                    supabase_client
                )

                # Refresh document list
                current_documents = get_existing_documents(
                    supabase_client
                )

                if current_documents:

                    print(
                        "\nThe document list has been refreshed."
                    )

                break

            # -------------------------------------------------
            # Delete document
            # -------------------------------------------------

            elif next_action == "4":

                current_documents = get_existing_documents(
                    supabase_client
                )

                delete_one_document(
                    supabase_client,
                    current_documents
                )

                # Refresh documents after deletion
                current_documents = get_existing_documents(
                    supabase_client
                )

                if not current_documents:

                    print(
                        "\nNo documents remain."
                    )

                    return "menu"

                # Remove deleted documents from current selection
                selected_documents = [
                    document["file_name"]
                    for document in current_documents
                    if document["file_name"]
                    in selected_documents
                ]

                # If all selected documents were deleted,
                # return to document selection.
                if not selected_documents:

                    print(
                        "\nYour previously selected documents "
                        "are no longer available."
                    )

                    new_selection = select_documents(
                        current_documents
                    )

                    if new_selection is None:

                        return "menu"

                    selected_documents = new_selection

                break

            # -------------------------------------------------
            # Main menu
            # -------------------------------------------------

            elif next_action == "5":

                return "menu"

            # -------------------------------------------------
            # Exit
            # -------------------------------------------------

            elif next_action == "6":

                return "exit"

            else:

                print(
                    "\nInvalid choice."
                )


# =============================================================
# PROCESS NEW DOCUMENT
# =============================================================

def process_new_document(
    supabase_client
):
    """
    Run the complete document ingestion pipeline:

        File
        ↓
        File type detection
        ↓
        Text extraction
        ↓
        Chunking
        ↓
        Embedding
        ↓
        Supabase insertion
    """

    file_path = input(
        "\nEnter the path to the document "
        "(or 'back'): "
    ).strip()

    if file_path.lower() == "back":

        return False

    if not file_path:

        print(
            "\nFile path cannot be empty."
        )

        return False

    # ---------------------------------------------------------
    # 1. Detect file type
    # ---------------------------------------------------------

    detection_result = detect_file_type(
        file_path
    )

    if not detection_result["success"]:

        print(
            detection_result
        )

        return False

    # ---------------------------------------------------------
    # 2. Extract document
    # ---------------------------------------------------------

    if detection_result["file_type"] == "pdf":

        extraction_result = extract_pdf_text(
            file_path
        )

        if not extraction_result["success"]:

            print(
                extraction_result
            )

            return False

        extracted_documents = extraction_result[
            "documents"
        ]

    else:

        print(
            f"\nFile type "
            f"'{detection_result['file_type']}' "
            "does not have an extraction pipeline yet."
        )

        return False

    # ---------------------------------------------------------
    # 3. Chunk document
    # ---------------------------------------------------------

    chunks = recursive_character_chunking(
        extracted_documents
    )

    if not chunks:

        print(
            "\nNo chunks were generated from the document."
        )

        return False

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------

    embedding_result = embed_chunks(
        chunks
    )

    if not embedding_result["success"]:

        print(
            embedding_result
        )

        return False

    # ---------------------------------------------------------
    # 5. Insert embeddings
    # ---------------------------------------------------------

    insertion_result = insert_documents(
        client=supabase_client,
        chunks=chunks,
        embeddings=embedding_result["embeddings"],
    )

    if not insertion_result["success"]:

        print(
            insertion_result
        )

        return False

    # ---------------------------------------------------------
    # 6. Pipeline result
    # ---------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        "DOCUMENT INGESTION COMPLETED"
    )

    print(
        "============================================================"
    )

    print(
        f"File type       : "
        f"{detection_result['file_type']}"
    )

    print(
        f"Pages           : "
        f"{len(extracted_documents)}"
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

    print(
        "============================================================"
    )

    return True


# =============================================================
# DELETE ONE DOCUMENT
# =============================================================

def delete_one_document(
    supabase_client,
    documents
):
    """
    Allow the user to select and delete one document
    from the Supabase vector database.
    """

    if not documents:

        print(
            "\nNo documents are currently stored "
            "in the vector database."
        )

        return False

    print(
        "\nExisting documents:\n"
    )

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

        selection = input(
            "Select a document number to delete "
            "(or 'back'): "
        ).strip()

        if selection.lower() == "back":

            return False

        try:

            selection = int(
                selection
            )

        except ValueError:

            print(
                "Please enter a valid number."
            )

            continue

        if 1 <= selection <= len(documents):

            break

        print(
            "Invalid document number."
        )

    selected_document = documents[
        selection - 1
    ]

    file_name = selected_document[
        "file_name"
    ]

    # ---------------------------------------------------------
    # 2. Confirmation
    # ---------------------------------------------------------

    print(
        f"\nSelected document: "
        f"{file_name}"
    )

    confirmation = input(
        "\nAre you sure you want to delete "
        "this document? (yes/no): "
    ).strip().lower()

    if confirmation not in (
        "yes",
        "y"
    ):

        print(
            "\nDelete operation cancelled."
        )

        return False

    # ---------------------------------------------------------
    # 3. Delete document
    # ---------------------------------------------------------

    result = delete_document(
        client=supabase_client,
        file_name=file_name,
    )

    if result["success"]:

        print(
            f"\nSuccessfully deleted: "
            f"{file_name}"
        )

        print(
            f"Deleted records: "
            f"{result.get('deleted_count', 0)}"
        )

        return True

    print(
        result
    )

    return False


# =============================================================
# DELETE ALL DOCUMENTS
# =============================================================

def delete_all_documents_from_database(
    supabase_client,
    documents
):
    """
    Delete all documents and their associated chunks
    from the Supabase vector database.
    """

    if not documents:

        print(
            "\nNo documents are currently stored "
            "in the vector database."
        )

        return False

    # ---------------------------------------------------------
    # 1. Display document count
    # ---------------------------------------------------------

    print(
        f"\nThis will permanently delete "
        f"all {len(documents)} document(s) "
        "and their stored chunks."
    )

    # ---------------------------------------------------------
    # 2. Confirmation
    # ---------------------------------------------------------

    confirmation = input(
        "\nAre you sure you want to delete "
        "ALL documents? (yes/no): "
    ).strip().lower()

    if confirmation not in (
        "yes",
        "y"
    ):

        print(
            "\nDelete operation cancelled."
        )

        return False

    # ---------------------------------------------------------
    # 3. Delete all documents
    # ---------------------------------------------------------

    result = delete_all_documents(
        client=supabase_client
    )

    if result["success"]:

        print(
            "\nSuccessfully deleted all documents."
        )

        print(
            f"Deleted records: "
            f"{result.get('deleted_count', 0)}"
        )

        return True

    print(
        result
    )

    return False


# =============================================================
# MAIN MENU
# =============================================================

def display_main_menu():
    """
    Display the main CLI menu.
    """

    print(
        "\n============================================================"
    )

    print(
        "                    AI RAG ASSIST"
    )

    print(
        "============================================================"
    )

    print(
        "1. Query existing document(s)"
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

    print(
        "5. Exit"
    )

    print(
        "============================================================"
    )


# =============================================================
# MAIN
# =============================================================

def main():

    # =========================================================
    # 1. Connect to Supabase
    # =========================================================

    supabase_result = connect_to_supabase()

    if not supabase_result["success"]:

        print(
            supabase_result
        )

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

        print(
            database_result
        )

        return

    # =========================================================
    # 3. Persistent application loop
    # =========================================================

    while True:

        # -----------------------------------------------------
        # Refresh document list every time the menu is shown
        # -----------------------------------------------------

        existing_documents = get_existing_documents(
            supabase_client
        )

        # -----------------------------------------------------
        # Main menu
        # -----------------------------------------------------

        display_main_menu()

        choice = input(
            "\nEnter your choice: "
        ).strip()

        # =====================================================
        # 1. QUERY
        # =====================================================

        if choice == "1":

            if not existing_documents:

                print(
                    "\nNo documents are currently "
                    "stored in the vector database."
                )

                continue

            result = query_documents_session(
                supabase_client=supabase_client,
                documents=existing_documents
            )

            if result == "exit":

                break

        # =====================================================
        # 2. ADD DOCUMENT
        # =====================================================

        elif choice == "2":

            success = process_new_document(
                supabase_client
            )

            if success:

                print(
                    "\nReturning to main menu..."
                )

        # =====================================================
        # 3. DELETE ONE DOCUMENT
        # =====================================================

        elif choice == "3":

            delete_one_document(
                supabase_client,
                existing_documents
            )

        # =====================================================
        # 4. DELETE ALL DOCUMENTS
        # =====================================================

        elif choice == "4":

            delete_all_documents_from_database(
                supabase_client,
                existing_documents
            )

        # =====================================================
        # 5. EXIT
        # =====================================================

        elif choice == "5":

            print(
                "\nExiting AI RAG Assist."
            )

            break

        # =====================================================
        # INVALID
        # =====================================================

        else:

            print(
                "\nInvalid choice. "
                "Please select an option from 1 to 5."
            )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()