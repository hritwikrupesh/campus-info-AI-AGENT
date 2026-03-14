import logging
from typing import List, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the embedding model.
    Utilizes sentence-transformers via LangChain's HuggingFaceEmbeddings wrapper.
    
    Returns:
        HuggingFaceEmbeddings: The loaded sentence-transformer embedding model.
    """
    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    try:
        # Use HuggingFaceEmbeddings to load sentence-transformers models natively for LangChain
        model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        logger.info("Successfully loaded the embedding model.")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model {MODEL_NAME}. Error: {e}")
        raise

def embed_documents(documents: List[Document]) -> List[List[float]]:
    """
    Accepts document chunks, converts their content into embeddings, and returns them.
    
    Args:
        documents (List[Document]): A list of Document objects (typically output from load_documents).
        
    Returns:
        List[List[float]]: A list of embeddings corresponding to the document chunks.
    """
    if not documents:
        logger.warning("No documents provided for embedding.")
        return []
    
    logger.info(f"Attempting to embed {len(documents)} document chunk(s)...")
    
    try:
        model = get_embedding_model()
        
        # Extract the string content from each LangChain Document object
        texts = [doc.page_content for doc in documents]
        
        # We use .embed_documents() from the Langchain embeddings interface
        embeddings = model.embed_documents(texts)
        
        logger.info(f"Successfully embedded {len(embeddings)} document chunk(s).")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error occurred during document embedding: {e}")
        raise

if __name__ == "__main__":
    # Test block
    from backend.rag.document_loader import load_documents
    docs = load_documents()
    if docs:
        embedded = embed_documents(docs)
        if embedded:
             print(f"Sample embedding vector (first 5 values): {embedded[0][:5]}...")
