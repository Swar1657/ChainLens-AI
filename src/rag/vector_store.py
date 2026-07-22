import os
import glob
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChromaDB persistence directory
PERSIST_DIR = "data/chromadb"
CORPUS_DIR = "data/corpus"

def get_embeddings():
    """Returns local FastEmbed embeddings. Does not require PyTorch or OpenAI keys."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def load_and_chunk_corpus() -> list[Document]:
    """Loads markdown files from the corpus directory and chunks them."""
    docs = []
    
    if not os.path.exists(CORPUS_DIR):
        print(f"Corpus directory {CORPUS_DIR} not found.")
        return docs

    md_files = glob.glob(f"{CORPUS_DIR}/*.md")
    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            filename = Path(file_path).name
            # Treat the whole file as a base document
            doc = Document(page_content=content, metadata={"source": filename})
            docs.append(doc)

    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n* ", "\n\n", "\n", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(docs)
    print(f"Loaded {len(md_files)} files. Split into {len(chunked_docs)} chunks.")
    return chunked_docs

def ingest_corpus():
    """Ingests the corpus into the persistent ChromaDB."""
    print("Ingesting RAG Corpus...")
    chunked_docs = load_and_chunk_corpus()
    
    if not chunked_docs:
        print("No documents to ingest.")
        return
        
    embeddings = get_embeddings()
    
    # Store into Chroma
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("Ingestion complete.")

def retrieve_context(query: str, k: int = 2) -> list[Document]:
    """Retrieves the top-k most relevant documents from the vector store."""
    if not os.path.exists(PERSIST_DIR):
        print("Warning: ChromaDB persist directory not found. Returning empty context.")
        return []
        
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    
    docs = vectorstore.similarity_search(query, k=k)
    return docs

def format_retrieved_context(docs: list[Document]) -> str:
    """Formats retrieved documents into a string for LLM context."""
    if not docs:
        return "No relevant operational guidelines found."
        
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        formatted.append(f"--- SOURCE: {source} ---\n{d.page_content}")
        
    return "\n\n".join(formatted)

if __name__ == "__main__":
    ingest_corpus()
