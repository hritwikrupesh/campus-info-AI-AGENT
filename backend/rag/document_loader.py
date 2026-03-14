import os
import glob
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for document chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    Assumes this file is located at `backend/rag/document_loader.py`.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

def load_documents() -> List[Document]:
    """
    Scan the 'data' directory for PDF files, load them, split them into chunks,
    and log the progress.
    
    Returns:
        List[Document]: A list of document chunks ready for embedding.
    """
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found at: {data_dir}")
        return []

    # Search for all PDF files in the data directory (and subdirectories)
    pdf_pattern = os.path.join(data_dir, '**', '*.pdf')
    pdf_files = glob.glob(pdf_pattern, recursive=True)
    
    if not pdf_files:
        logger.info("No PDF files found in the data directory.")
        return []

    num_pdfs = len(pdf_files)
    logger.info(f"Found {num_pdfs} PDF(s) to process.")
    
    documents = []
    
    # Load each PDF using LangChain's PyPDFLoader
    for pdf_path in pdf_files:
        try:
            logger.debug(f"Loading document: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {e}")

    logger.info(f"Successfully loaded {num_pdfs} PDF document(s).")
    
    # Split documents into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    logger.info(f"Splitting documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"Created {len(chunks)} document chunk(s) from {num_pdfs} document(s).")
    
    return chunks

if __name__ == "__main__":
    # Optional execution block to test loading behavior directly
    chunks = load_documents()
    if chunks:
        print(f"Sample chunk:\n{chunks[0].page_content[:200]}...")
