from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import bs4

# GLOBALS
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

def setup_pipeline_from_pdf(file_path, persist_path="db"):
    global db, chain
    docs = load_from_pdf(file_path)
    chunks = split_documents(docs)
    db = create_vector_store(chunks, persist_path)
    chain = build_llm_chain(db)

def load_existing_pipeline(persist_path="db"):
    global db, chain
    db = load_vector_store(persist_path)
    chain = build_llm_chain(db)

def ask(query):
    global chain
    if not chain:
        raise ValueError("Pipeline not initialized. Call setup_pipeline_from_url() or load_existing_pipeline() first.")
    return chain.invoke({"input": query})["answer"]

# ===========================
# Direct Run (optional demo)
# ===========================

if __name__ == "__main__":
 
    
    setup_pipeline_from_url("https://en.wikipedia.org/wiki/Samantha_Ruth_Prabhu")
    print(ask("What awards has Sam won?"))
    print(ask("What movies has she worked with?"))
    print(ask("Tell me about Elon Musk."))  # This should return "out of context"
