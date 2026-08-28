# AI RAG Assist

> ⚠️ **This README is a work in progress and will be updated.**

A Retrieval-Augmented Generation (RAG) pipeline that lets you ingest documents, store their embeddings in a vector database, and query them using an LLM for context-aware answers.

## Tech Stack

| Layer            | Technology                  |
| ---------------- | --------------------------- |
| Language         | Python 3.14+                |
| LLM              | Groq                       |
| Embeddings       | Sentence Transformers       |
| Vector Database  | Supabase (pgvector)         |
| Orchestration    | LangChain                   |
| Document Parsing | PyPDF, Unstructured         |
| Package Manager  | uv                          |

## Project Structure

```
AI_Rag_Assist/
├── main.py                # CLI entry point
├── ingestion/             # File type detection & text extraction
├── chunking/              # Recursive character text splitting
├── embedding/             # Sentence-transformer embedding generation
├── vector_db/             # Supabase client, insert, query, list operations
├── llm/                   # Groq LLM answer generation
├── TestData/              # Sample documents for testing
├── logs/                  # Application logs
├── version_0/             # Initial prototype / reference
├── pyproject.toml         # Project metadata & dependencies
└── uv.lock               # Locked dependency versions
```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd AI_Rag_Assist

# Install dependencies (using uv)
uv sync

# Configure environment variables
# Create a .env file with your Supabase and Groq credentials
```

## Usage

```bash
uv run main.py
```

The CLI will prompt you to:

1. **Query an existing document** — select a previously ingested document and ask questions against it.
2. **Add a new document** — provide a file path to run the full ingestion pipeline (detect → extract → chunk → embed → store), then query it.

## Pipeline Overview

```
Document → File Detection → Text Extraction → Chunking → Embedding → Supabase (pgvector)
                                                                            ↓
                                                        User Query → Vector Search → LLM → Answer
```

---

*This README will be expanded with detailed API docs, configuration reference, and contribution guidelines.*
