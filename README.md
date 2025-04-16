# RAG AI Assistant Flask Application

This Flask application provides an AI assistant that can extract and query information from either Wikipedia articles or PDF documents using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- Upload PDF documents or provide Wikipedia URLs for knowledge extraction
- Ask questions about the content using natural language
- Dark-themed user interface with responsive design
- Session management to maintain context between questions

## Prerequisites

- Python 3.8 or higher
- Ollama running locally with the llama3.2 model installed
  - Install from [ollama.ai](https://ollama.ai/)
  - Run `ollama pull llama3.2` to download the model
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```
git clone https://github.com/MONARCH1108/AI_Rag_Assist
cd flask_rag_app
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Running the Application

1. Make sure Ollama is running in the background with the llama3.2 model installed.

2. Start the Flask application:
```
python app.py
```

3. Open your web browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. **Wikipedia Mode**:
   - Enter a Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Artificial_intelligence)
   - Click "Load Article" and wait for processing to complete

2. **PDF Mode**:
   - Click the "PDF" tab
   - Select a PDF file from your computer
   - Click "Upload PDF" and wait for processing to complete

3. **Asking Questions**:
   - Once content is loaded, the chat input will be enabled
   - Type your question and press Enter or click the send button
   - The AI will respond based on the content of the loaded document

4. **Starting Over**:
   - Click "Clear Session" to reset the application and load new content

## Technical Details

This application uses:
- LangChain for document processing and the RAG pipeline
- HuggingFace embeddings (sentence-transformers/paraphrase-MiniLM-L6-v2)
- Ollama's local LLM (llama3.2) for answering questions
- ChromaDB for vector storage
- Flask for the web server

## Notes

- Processing large documents may take time, especially on systems with limited resources
- The quality of answers depends on both the content provided and the capabilities of the llama3.2 model
- For best results, ask specific questions related to the content of the loaded document
