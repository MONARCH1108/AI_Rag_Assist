from dotenv import load_dotenv
load_dotenv()
from utils.logger import logger
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

    logger.info(
        "Retrieving existing documents from Supabase"
    )
    documents_result = list_documents(
        client=supabase_client
    )
    if not documents_result["success"]:
        logger.error(
            "Failed to retrieve existing documents"
        )
        print(
            "\nFailed to retrieve documents."
        )
        print(
            documents_result
        )
        return []
    documents = documents_result[
        "documents"
    ]
    logger.info(
        "Retrieved %s existing document(s)",
        len(documents)
    )
    return documents


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

    logger.info(
        "Starting document selection"
    )
    if not documents:
        logger.info(
            "Document selection requested, but no documents exist"
        )
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
            logger.info(
                "Document selection cancelled by user"
            )
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
            logger.warning(
                "Invalid document selection input: %s",
                selection
            )
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
            logger.warning(
                "Document selection contains invalid index: %s",
                selected_indexes
            )
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
    logger.info(
        "Selected %s document(s): %s",
        len(selected_documents),
        selected_documents
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

    logger.info(
        "Starting RAG query execution"
    )

    if not query or not query.strip():

        logger.warning(
            "Empty query received"
        )

        print(
            "\nQuery cannot be empty."
        )

        return False

    logger.info(
        "Executing query against %s selected document(s)",
        len(selected_documents)
    )

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

        logger.error(
            "Vector search failed for query"
        )

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

        logger.info(
            "No relevant chunks found for query"
        )

        print(
            "\nNo relevant chunks were found "
            "in the selected document(s)."
        )

        return True

    logger.info(
        "Retrieved %s relevant chunks",
        len(query_result["results"])
    )

    # ---------------------------------------------------------
    # 3. Generate answer
    # ---------------------------------------------------------

    llm_result = generate_answer(
        question=query,
        retrieved_chunks=query_result["results"],
    )

    if not llm_result["success"]:

        logger.error(
            "LLM generation failed"
        )

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

    logger.info(
        "RAG query completed successfully"
    )

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

    logger.info(
        "Starting persistent query session"
    )

    # ---------------------------------------------------------
    # Initial document selection
    # ---------------------------------------------------------

    selected_documents = select_documents(
        documents
    )

    if selected_documents is None:

        logger.info(
            "Query session cancelled during initial selection"
        )

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

            logger.info(
                "Returning from query session to main menu"
            )

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

                logger.info(
                    "User selected another query"
                )

                break

            # -------------------------------------------------
            # Change document selection
            # -------------------------------------------------

            elif next_action == "2":

                logger.info(
                    "User selected change document selection"
                )

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

                logger.info(
                    "User selected add new document"
                )

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

                logger.info(
                    "User selected delete document"
                )

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

                logger.info(
                    "Returning to main menu from query session"
                )

                return "menu"

            # -------------------------------------------------
            # Exit
            # -------------------------------------------------

            elif next_action == "6":

                logger.info(
                    "User selected application exit"
                )

                return "exit"

            else:

                logger.warning(
                    "Invalid query session action: %s",
                    next_action
                )

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

    logger.info(
        "Starting document ingestion pipeline"
    )

    file_path = input(
        "\nEnter the path to the document "
        "(or 'back'): "
    ).strip()

    if file_path.lower() == "back":

        logger.info(
            "Document ingestion cancelled by user"
        )

        return False

    if not file_path:

        logger.warning(
            "Empty document path received"
        )

        print(
            "\nFile path cannot be empty."
        )

        return False

    logger.info(
        "Processing document: %s",
        file_path
    )

    # ---------------------------------------------------------
    # 1. Detect file type
    # ---------------------------------------------------------

    detection_result = detect_file_type(
        file_path
    )

    if not detection_result["success"]:

        logger.error(
            "File type detection failed: %s",
            file_path
        )

        print(
            detection_result
        )

        return False

    # ---------------------------------------------------------
    # 2. Extract document
    # ---------------------------------------------------------

    if detection_result["file_type"] == "pdf":

        logger.info(
            "Starting PDF extraction: %s",
            detection_result["file_name"]
        )

        extraction_result = extract_pdf_text(
            file_path
        )

        if not extraction_result["success"]:

            logger.error(
                "PDF extraction failed: %s",
                detection_result["file_name"]
            )

            print(
                extraction_result
            )

            return False

        extracted_documents = extraction_result[
            "documents"
        ]

    else:

        logger.warning(
            "Unsupported ingestion pipeline for file type: %s",
            detection_result["file_type"]
        )

        print(
            f"\nFile type "
            f"'{detection_result['file_type']}' "
            "does not have an extraction pipeline yet."
        )

        return False

    # ---------------------------------------------------------
    # 3. Chunk document
    # ---------------------------------------------------------

    logger.info(
        "Starting document chunking"
    )

    chunks = recursive_character_chunking(
        extracted_documents
    )

    if not chunks:

        logger.warning(
            "No chunks were generated from document: %s",
            detection_result["file_name"]
        )

        print(
            "\nNo chunks were generated from the document."
        )

        return False

    logger.info(
        "Document chunking completed: %s chunks created",
        len(chunks)
    )

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------

    logger.info(
        "Starting embedding generation"
    )

    embedding_result = embed_chunks(
        chunks
    )

    if not embedding_result["success"]:

        logger.error(
            "Embedding generation failed"
        )

        print(
            embedding_result
        )

        return False

    # ---------------------------------------------------------
    # 5. Insert embeddings
    # ---------------------------------------------------------

    logger.info(
        "Starting Supabase vector insertion"
    )

    insertion_result = insert_documents(
        client=supabase_client,
        chunks=chunks,
        embeddings=embedding_result["embeddings"],
    )

    if not insertion_result["success"]:

        logger.error(
            "Supabase vector insertion failed"
        )

        print(
            insertion_result
        )

        return False

    # ---------------------------------------------------------
    # 6. Pipeline result
    # ---------------------------------------------------------

    logger.info(
        "Document ingestion pipeline completed successfully: %s",
        detection_result["file_name"]
    )

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

    logger.info(
        "Starting single document deletion"
    )

    if not documents:

        logger.info(
            "Delete requested, but no documents are stored"
        )

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

            logger.info(
                "Document deletion cancelled by user"
            )

            return False

        try:

            selection = int(
                selection
            )

        except ValueError:

            logger.warning(
                "Invalid delete selection input: %s",
                selection
            )

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

        logger.info(
            "Document deletion cancelled during confirmation: %s",
            file_name
        )

        print(
            "\nDelete operation cancelled."
        )

        return False

    # ---------------------------------------------------------
    # 3. Delete document
    # ---------------------------------------------------------

    logger.info(
        "Deleting document: %s",
        file_name
    )

    result = delete_document(
        client=supabase_client,
        file_name=file_name,
    )

    if result["success"]:

        logger.info(
            "Document deleted successfully: %s",
            file_name
        )

        print(
            f"\nSuccessfully deleted: "
            f"{file_name}"
        )

        print(
            f"Deleted records: "
            f"{result.get('deleted_count', 0)}"
        )

        return True

    logger.error(
        "Failed to delete document: %s",
        file_name
    )

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

    logger.info(
        "Starting delete-all-documents operation"
    )

    if not documents:

        logger.info(
            "Delete-all requested, but no documents are stored"
        )

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

        logger.info(
            "Delete-all operation cancelled by user"
        )

        print(
            "\nDelete operation cancelled."
        )

        return False

    # ---------------------------------------------------------
    # 3. Delete all documents
    # ---------------------------------------------------------

    logger.warning(
        "Deleting ALL documents from Supabase"
    )

    result = delete_all_documents(
        client=supabase_client
    )

    if result["success"]:

        logger.info(
            "Successfully deleted all documents: %s records",
            result.get("deleted_count", 0)
        )

        print(
            "\nSuccessfully deleted all documents."
        )

        print(
            f"Deleted records: "
            f"{result.get('deleted_count', 0)}"
        )

        return True

    logger.error(
        "Failed to delete all documents"
    )

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

    logger.info(
        "============================================================"
    )

    logger.info(
        "Starting AI RAG Assist"
    )

    logger.info(
        "============================================================"
    )

    # =========================================================
    # 1. Connect to Supabase
    # =========================================================

    logger.info(
        "Connecting to Supabase"
    )

    supabase_result = connect_to_supabase()

    if not supabase_result["success"]:

        logger.error(
            "Supabase connection failed"
        )

        print(
            supabase_result
        )

        return

    supabase_client = supabase_result[
        "client"
    ]

    logger.info(
        "Supabase connection established"
    )

    # =========================================================
    # 2. Verify database
    # =========================================================

    logger.info(
        "Verifying Supabase database"
    )

    database_result = verify_supabase_database(
        supabase_client
    )

    if not database_result["success"]:

        logger.error(
            "Supabase database verification failed"
        )

        print(
            database_result
        )

        return

    logger.info(
        "Supabase database verification completed"
    )

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

        logger.info(
            "Main menu choice selected: %s",
            choice
        )

        # =====================================================
        # 1. QUERY
        # =====================================================

        if choice == "1":

            if not existing_documents:

                logger.info(
                    "Query option selected, but no documents exist"
                )

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

                logger.info(
                    "Application exit requested from query session"
                )

                break

        # =====================================================
        # 2. ADD DOCUMENT
        # =====================================================

        elif choice == "2":

            logger.info(
                "Add-document option selected"
            )

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

            logger.info(
                "Delete-document option selected"
            )

            delete_one_document(
                supabase_client,
                existing_documents
            )

        # =====================================================
        # 4. DELETE ALL DOCUMENTS
        # =====================================================

        elif choice == "4":

            logger.warning(
                "Delete-all-documents option selected"
            )

            delete_all_documents_from_database(
                supabase_client,
                existing_documents
            )

        # =====================================================
        # 5. EXIT
        # =====================================================

        elif choice == "5":

            logger.info(
                "AI RAG Assist shutting down"
            )

            print(
                "\nExiting AI RAG Assist."
            )

            break

        # =====================================================
        # INVALID
        # =====================================================

        else:

            logger.warning(
                "Invalid main menu choice: %s",
                choice
            )

            print(
                "\nInvalid choice. "
                "Please select an option from 1 to 5."
            )

    logger.info(
        "AI RAG Assist stopped"
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()