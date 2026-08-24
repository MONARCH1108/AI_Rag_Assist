from flask import Flask, render_template, request, jsonify, session
import os
from werkzeug.utils import secure_filename
import tempfile

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import bs4

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Required for session management
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Global variables
db = None
chain = None

# ===========================
# Loaders
# ===========================

def load_from_url(url):
    loader = WebBaseLoader(
        web_path=url,
        bs_kwargs=dict(parse_only=bs4.SoupStrainer("p"))
    )
    return loader.load()

def load_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# ===========================
# Splitter
# ===========================

def split_documents(docs, chunk_size=1000, chunk_overlap=10):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# ===========================
# Vector DB
# ===========================

def create_vector_store(documents, persist_path="db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_path)
    return vectordb

def load_vector_store(persist_path="db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    return vectordb

# ===========================
# LLM + Retrieval Chain
# ===========================

def build_llm_chain(vectordb):
    model = OllamaLLM(model="llama3.2")

    prompt = ChatPromptTemplate.from_template(
        """
        1. You are an AI assistant who only answers from the given context.
        2. Do not answer using outside knowledge.
        3. Elaborate using only context.
        4. If asked out-of-context, reply "out of context".

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = vectordb.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

# ===========================
# Setup & Query Functions
# ===========================

def setup_pipeline_from_url(url, persist_path="db"):
    global db, chain
    docs = load_from_url(url)
    chunks = split_documents(docs)
    db = create_vector_store(chunks, persist_path)
    chain = build_llm_chain(db)
    return "Pipeline set up successfully from URL"

def setup_pipeline_from_pdf(file_path, persist_path="db"):
    global db, chain
    docs = load_from_pdf(file_path)
    chunks = split_documents(docs)
    db = create_vector_store(chunks, persist_path)
    chain = build_llm_chain(db)
    return "Pipeline set up successfully from PDF"

def load_existing_pipeline(persist_path="db"):
    global db, chain
    db = load_vector_store(persist_path)
    chain = build_llm_chain(db)
    return "Existing pipeline loaded"

def ask(query):
    global chain
    if not chain:
        return "Pipeline not initialized. Please upload a PDF or enter a Wikipedia URL first."
    return chain.invoke({"input": query})["answer"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ===========================
# Flask Routes
# ===========================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_wiki', methods=['POST'])
def process_wiki():
    wiki_url = request.form.get('wiki_url')
    if not wiki_url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        result = setup_pipeline_from_url(wiki_url)
        session['source_type'] = 'wiki'
        session['source'] = wiki_url
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = setup_pipeline_from_pdf(filepath)
            session['source_type'] = 'pdf'
            session['source'] = filename
            return jsonify({"message": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        answer = ask(query)
        return jsonify({
            "answer": answer, 
            "source_type": session.get('source_type', ''),
            "source": session.get('source', '')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    session.clear()
    global db, chain
    db = None
    chain = None
    return jsonify({"message": "Session cleared"})

if __name__ == '__main__':
    app.run(debug=True)
