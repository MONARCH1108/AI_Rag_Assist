# AI RAG Assist — Architecture & Process Flows

> Auto-generated flow diagrams. Reflects the current state of the codebase.

---

## 1. Overall System Architecture

```mermaid
graph TD
    User(["User"]) -->|CLI| CLI["cli-rag-assist.py"]
    User -->|HTTP| API["main.py (FastAPI)"]

    CLI --> IngestionPipeline["Ingestion Pipeline"]
    CLI --> QueryPipeline["Query Pipeline"]
    CLI --> DeleteOps["Delete Operations"]

    API --> IngestionPipeline
    API --> QueryPipeline
    API --> ListDocs["List Documents"]

    IngestionPipeline --> Ingest["ingestion/"]
    IngestionPipeline --> Chunk["chunking/"]
    IngestionPipeline --> Embed["embedding/"]
    IngestionPipeline --> Insert["vector_db/insert_embeddings.py"]

    QueryPipeline --> EmbedQuery["embedding/ (query)"]
    QueryPipeline --> VectorSearch["vector_db/query_vectors.py"]
    QueryPipeline --> LLM["llm/groq.py"]

    Insert --> Supabase[("Supabase pgvector")]
    VectorSearch --> Supabase
    ListDocs --> Supabase
    DeleteOps --> Supabase
```

The system has two interfaces: a **CLI** (`cli-rag-assist.py`) and a **FastAPI backend** (`main.py`). Both share the same underlying modules for ingestion, embedding, vector storage, retrieval, and LLM generation. Supabase with pgvector serves as the single vector database.

---

## 2. Document Ingestion Pipeline

```mermaid
flowchart TD
    A["Receive Document (file path or upload)"] --> B["detect_file_type()"]
    B -->|Unsupported| X1["Return UNSUPPORTED_FILE_TYPE error"]
    B -->|PDF detected| C["extract_pdf_text()"]
    C -->|Extraction failed| X2["Return PDF_EXTRACTION_ERROR"]
    C -->|Success| D["Page-by-page LangChain Documents with metadata"]
    D --> E["recursive_character_chunking()"]
    E -->|No chunks| X3["Return NO_CHUNKS_GENERATED error"]
    E -->|Chunks created| F["embed_chunks()"]
    F -->|Model load failed| X4["Return EMBEDDING_MODEL_ERROR"]
    F -->|Encoding complete| G["384-dim normalized embeddings"]
    G --> H["insert_documents()"]
    H -->|Batch insert (size=25)| I[("Supabase documents table")]
    H -->|Insert failed| X5["Return SUPABASE_INSERTION_ERROR"]
    I --> J["Return success with stats"]
```

Both the CLI and FastAPI pipelines follow identical stages: **detect → extract → chunk → embed → insert**. The FastAPI path additionally validates the guest session and handles temporary file management.

---

## 3. RAG Query Pipeline

```mermaid
flowchart TD
    A["User submits query"] --> B["Embed query with all-MiniLM-L6-v2"]
    B --> C["Call Supabase match_documents RPC"]
    C -->|"threshold=0.20, top_k=5"| D{Matching chunks found?}
    D -->|No| E["Return: No relevant chunks found"]
    D -->|Yes| F["Format retrieved context with scores and metadata"]
    F --> G["Build RAG prompt (system + user)"]
    G --> H["Send to Groq LLM (temperature=0)"]
    H --> I{Answer received?}
    I -->|Empty| J["Return EMPTY_LLM_RESPONSE error"]
    I -->|Valid| K["Return grounded answer"]
```

The query pipeline embeds the user's question using the same model used during ingestion, searches Supabase via the `match_documents` PostgreSQL RPC function, and passes the top-k results to Groq for grounded answer generation.

---

## 4. FastAPI Request Lifecycle (API Endpoints)

```mermaid
flowchart TD
    subgraph "POST /run-pipeline"
        A1["Receive file upload(s)"] --> A2["Connect to Supabase"]
        A2 --> A3["Validate X-User-ID guest session"]
        A3 -->|Invalid| A4["Return session error"]
        A3 -->|Valid| A5["For each file: run ingestion pipeline"]
        A5 --> A6["Cleanup temp files"]
        A6 --> A7["Return per-file results"]
    end

    subgraph "GET /documents"
        B1["Receive request"] --> B2["Connect to Supabase"]
        B2 --> B3["Validate X-User-ID guest session"]
        B3 -->|Invalid| B4["Return session error"]
        B3 -->|Valid| B5["list_documents(user_id)"]
        B5 --> B6["Return user's documents"]
    end

    subgraph "GET /health"
        C1["Receive request"] --> C2["check_supabase_health()"]
        C2 --> C3["check_groq_health()"]
        C3 --> C4["Return health status"]
    end
```

All data-mutating endpoints validate the guest session via the `guest_sessions` table before proceeding. The health endpoint checks both Supabase and Groq dependencies.

---

## 5. Guest Session Validation

```mermaid
flowchart TD
    A["Receive X-User-ID header"] --> B{User ID provided?}
    B -->|No| C["Return USER_ID_MISSING"]
    B -->|Yes| D["Query guest_sessions table"]
    D --> E{Session found?}
    E -->|No| F["Return INVALID_USER_ID"]
    E -->|Yes| G{Session expired?}
    G -->|Yes| H["Return USER_SESSION_EXPIRED"]
    G -->|No| I["Update last_activity_at"]
    I --> J["Return success with user_id"]
```

---

## 6. Supabase Connection with Retry Logic

```mermaid
flowchart TD
    A["Read SUPABASE_URL and SUPABASE_KEY"] --> B{Credentials present?}
    B -->|No| C["Return configuration error"]
    B -->|Yes| D["Attempt 1: create_client + verify table"]
    D -->|Success| E["Return connected client"]
    D -->|Failure| F["Wait 2s"]
    F --> G["Attempt 2: retry"]
    G -->|Success| E
    G -->|Failure| H["Wait 4s"]
    H --> I["Attempt 3: retry"]
    I -->|Success| E
    I -->|Failure| J["Return SUPABASE_CONNECTION_ERROR"]
```

The `connect_to_supabase()` function implements exponential backoff (2s, 4s) across up to 3 connection attempts.

---

## 7. CLI Interactive Session Flow

```mermaid
flowchart TD
    A["Start cli-rag-assist.py"] --> B["Connect to Supabase"]
    B --> C["Verify database"]
    C --> D["Main Menu Loop"]

    D -->|"1. Query"| E["Select document(s)"]
    E --> F["Enter query"]
    F --> G["Execute RAG pipeline"]
    G --> H["Post-query menu"]
    H -->|Another query| F
    H -->|Change docs| E
    H -->|Add doc| I["Run ingestion pipeline"]
    H -->|Delete doc| J["Delete selected document"]
    H -->|Main menu| D
    H -->|Exit| K["Shutdown"]

    D -->|"2. Add document"| I
    I --> D

    D -->|"3. Delete one"| J
    J --> D

    D -->|"4. Delete all"| L["Confirm + delete all"]
    L --> D

    D -->|"5. Exit"| K
```

---

## 8. Document Deletion Flow

```mermaid
flowchart TD
    subgraph "Single Document Deletion"
        A["Select document by number"] --> B["Confirm deletion"]
        B -->|No| C["Cancel"]
        B -->|Yes| D["delete_document(user_id, file_name)"]
        D --> E["DELETE FROM documents WHERE user_id AND file_name"]
        E --> F["Return deleted count"]
    end

    subgraph "Bulk Document Deletion"
        G["Display document count"] --> H["Confirm 'yes'"]
        H -->|No| I["Cancel"]
        H -->|Yes| J["delete_all_documents(user_id)"]
        J --> K["DELETE FROM documents WHERE user_id"]
        K --> L["Return deleted count"]
    end
```

Both deletion functions scope operations to the specific user_id, ensuring user-level data isolation.

---

## 9. Embedding Generation Flow

```mermaid
flowchart TD
    A["Receive LangChain Document chunks"] --> B["Validate chunks are non-empty"]
    B --> C["Load / retrieve cached model"]
    C -->|"_MODEL_CACHE miss"| D["SentenceTransformer('all-MiniLM-L6-v2')"]
    D --> E["Cache model in _MODEL_CACHE"]
    C -->|"_MODEL_CACHE hit"| F["Use cached model"]
    E --> F
    F --> G["Extract page_content from each chunk"]
    G --> H["model.encode(texts, normalize=True)"]
    H --> I["Return 384-dim embedding vectors"]
```

The model is loaded once and cached in-memory for the lifetime of the process, avoiding redundant weight loading.

---

## 10. Health Check Flow

```mermaid
flowchart TD
    A["GET /health"] --> B["check_supabase_health()"]
    B --> B1["Validate credentials"]
    B1 --> B2["Create/reuse Supabase client"]
    B2 --> B3["SELECT id FROM documents LIMIT 1"]
    B3 --> B4["Return Supabase health status"]

    A --> C["check_groq_health()"]
    C --> C1["Validate GROQ_API_KEY + MODEL_NAME"]
    C1 --> C2["Initialize Groq client"]
    C2 --> C3["Send test completion ('hi', max_tokens=5)"]
    C3 --> C4["Return Groq health status"]

    B4 --> D{Both healthy?}
    C4 --> D
    D -->|Yes| E["status: healthy"]
    D -->|No| F["status: unhealthy"]
```
