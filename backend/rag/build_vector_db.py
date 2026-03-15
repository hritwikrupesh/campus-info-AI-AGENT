import os
import json
import logging
import shutil
from typing import List

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
RAW_DATA_DIR = "data/raw"
VECTOR_DB_DIR = "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format to match example output
)
logger = logging.getLogger(__name__)


def load_documents() -> List[Document]:
    """
    Read every JSON file from data/raw/ and convert them into LangChain Document objects.
    """
    logger.info(f"Loading documents from {RAW_DATA_DIR}/")
    documents = []
    
    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"Directory {RAW_DATA_DIR} does not exist.")
        return documents

    json_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
    
    for file_name in json_files:
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            content = data.get("content", "").strip()
            url = data.get("url", "")
            title = data.get("title", "")
            
            # Skip safely if missing fields or empty content
            if not content:
                continue
                
            doc = Document(
                page_content=content,
                metadata={
                    "url": url,
                    "title": title
                }
            )
            documents.append(doc)
            
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read or parse {file_name}: {e}. Skipping.")

    logger.info(f"Loaded {len(documents)} documents\n")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks suitable for RAG retrieval.
    """
    logger.info("Splitting documents into chunks")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} text chunks\n")
    
    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return the sentence-transformers HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def build_vector_database(chunks: List[Document], embeddings: HuggingFaceEmbeddings):
    """
    Create a fresh Chroma vector store and persist it locally.
    """
    logger.info("Building Chroma vector database")
    
    # If a previous vector database exists, delete it first
    if os.path.exists(VECTOR_DB_DIR):
        try:
            shutil.rmtree(VECTOR_DB_DIR)
        except Exception as e:
            logger.error(f"Could not delete existing vector database at {VECTOR_DB_DIR}: {e}")
            raise

    logger.info("Embedding chunks...")
    
    try:
        # Chroma dynamically accepts chunks and creates the local index in vector_db/
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        
        logger.info(f"\nVector database successfully created at {VECTOR_DB_DIR}/")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error building vector database: {e}")
        raise


def run_pipeline():
    """
    Execute the entire Data -> Vector Store RAG pipeline.
    """
    # 1. Load JSON documents
    documents = load_documents()
    if not documents:
        logger.error("No documents to process. Aborting.")
        return
        
    # 2. Split them into chunks
    chunks = split_documents(documents)
    if not chunks:
        logger.error("No chunks were created. Aborting.")
        return
        
    # 3. Generate embeddings interface
    embeddings = create_embeddings()
    
    # 4 & 5. Build and persist the Chroma vector database
    build_vector_database(chunks, embeddings)


if __name__ == "__main__":
    run_pipeline()
