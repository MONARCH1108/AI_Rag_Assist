# AI RAG Assist — Directory Structure

> Auto-generated repository map. Reflects the current state of the codebase.

```
AI_Rag_Assist/
├── main.py
│   ├── validate_guest_session()       — Validates a guest user's session against the guest_sessions table and updates last_activity_at.
│   ├── health_check()                 — GET /health — Reports overall service health (Supabase + Groq).
│   ├── upload_documents()             — POST /run-pipeline — Accepts PDF uploads, runs the full ingestion pipeline per file, returns per-file results.
│   └── get_documents()                — GET /documents — Lists unique documents belonging to the authenticated guest user.
│
├── cli-rag-assist.py
│   ├── get_existing_documents()       — Retrieves unique document list from Supabase for the CLI session.
│   ├── select_documents()             — Interactive multi-document selector supporting single, comma-separated, and "all" input.
│   ├── execute_query()                — Runs the full RAG pipeline: embed query → vector search → Groq LLM → display answer.
│   ├── query_documents_session()      — Persistent query session with post-query menu (re-query, change docs, add, delete, exit).
│   ├── process_new_document()         — CLI document ingestion pipeline: detect → extract → chunk → embed → insert.
│   ├── delete_one_document()          — Interactive single-document deletion with confirmation.
│   ├── delete_all_documents_from_database() — Interactive bulk document deletion with confirmation.
│   ├── display_main_menu()            — Renders the main CLI menu.
│   └── main()                         — Application entry point: connects to Supabase, verifies DB, starts interactive loop.
│
├── ingestion/
│   ├── text_ingestion.py
│   │   └── detect_file_type()         — Validates file existence and maps extension to document type (pdf, doc, csv, etc.).
│   └── pdf_text_extraction.py
│       └── extract_pdf_text()         — Extracts text page-by-page from a PDF using PyPDF into LangChain Document objects with metadata.
│
├── chunking/
│   └── recursive_character_text_splitter.py
│       └── recursive_character_chunking() — Splits LangChain Documents using RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200).
│
├── embedding/
│   └── sentence_transformer.py
│       ├── _get_embedding_model()     — Loads and caches the Sentence Transformer model (all-MiniLM-L6-v2) with optional HF token.
│       └── embed_chunks()             — Generates normalized 384-dim vector embeddings for document chunks.
│
├── vector_db/
│   ├── supabase_client.py
│   │   └── connect_to_supabase()      — Creates a Supabase client with exponential-backoff retries (max 3 attempts) and verifies connectivity.
│   ├── verify_database.py
│   │   └── verify_supabase_database() — Validates that the Supabase documents table exists and is accessible.
│   ├── insert_embeddings.py
│   │   └── insert_documents()         — Inserts embedded chunks into Supabase in batches of 25 with user_id scoping and validation.
│   ├── list_documents.py
│   │   └── list_documents()           — Queries Supabase for all records belonging to a user and extracts unique documents by file_name/source.
│   └── query_vectors.py
│       └── query_vectors()            — Embeds a query, calls Supabase match_documents RPC (pgvector), and returns top-k results with scores.
│
├── delete/
│   └── delete_files.py
│       ├── delete_document()          — Deletes all chunks for a given file_name and user_id from Supabase.
│       └── delete_all_documents()     — Deletes all document chunks belonging to a user from Supabase.
│
├── llm/
│   ├── prompts.py
│   │   ├── RAG_SYSTEM_PROMPT          — Strict RAG system prompt enforcing context-only answers and zero hallucinations.
│   │   └── RAG_USER_PROMPT            — Template injecting retrieved context and user question into the LLM prompt.
│   └── groq.py
│       └── generate_answer()          — Formats retrieved chunks as context, calls Groq LLM (temperature=0), returns grounded answer.
│
├── utils/
│   ├── logger.py
│   │   └── logger                     — Configures the ai_rag_assist logger with console and optional file output (logs/ingestion.log).
│   ├── supabase_service.py
│   │   ├── get_supabase_client()      — Singleton Supabase client factory used by the health check service.
│   │   ├── check_supabase_health()    — Verifies Supabase credentials, client creation, DB reachability, and documents table access.
│   │   └── reset_supabase_client()    — Resets the cached Supabase client (for testing).
│   └── groq_service.py
│       └── check_groq_health()        — Validates Groq API key, model config, client init, connectivity, and model access via a test completion.
│
├── version_0/                          — Initial prototype / reference implementation.
│   ├── app.py
│   │   ├── load_from_url()            — Loads web page content using LangChain WebBaseLoader.
│   │   ├── load_from_pdf()            — Loads PDF using LangChain PyPDFLoader.
│   │   ├── split_documents()          — Splits documents using RecursiveCharacterTextSplitter.
│   │   ├── create_vector_store()      — Creates a Chroma vector store from documents.
│   │   ├── load_vector_store()        — Loads an existing Chroma vector store from disk.
│   │   ├── build_llm_chain()          — Builds an Ollama LLM retrieval chain.
│   │   ├── setup_pipeline_from_url()  — End-to-end pipeline setup from a URL.
│   │   ├── setup_pipeline_from_pdf()  — End-to-end pipeline setup from a PDF.
│   │   ├── load_existing_pipeline()   — Loads an existing pipeline from persisted vector store.
│   │   └── ask()                      — Queries the retrieval chain and returns the answer.
│   ├── Flask_app.py                   — Flask web application wrapping the version_0 pipeline.
│   ├── templates/
│   │   └── index.html                 — HTML template for the Flask web UI.
│   ├── requirements.txt               — Version 0 dependencies.
│   ├── README.md                      — Version 0 documentation.
│   └── LICENSE                        — Project license.
│
├── TestData/
│   ├── 623-Article Text-620-1-10-20080128.pdf
│   └── OHAE_2003_Amarna.pdf
│
├── logs/                               — Application log output directory.
├── documents/
│   └── PROMPT.md                      — Antigravity documentation workflow instructions.
│
├── .env                               — Environment variables (Supabase, Groq, HF credentials).
├── .gitignore                         — Git exclusion rules.
├── .python-version                    — Python version pin.
├── pyproject.toml                     — Project metadata and dependency declarations (uv).
├── uv.lock                            — Locked dependency versions.
└── README.md                          — Project README (work in progress).
```
