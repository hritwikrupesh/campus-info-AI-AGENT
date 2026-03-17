import os
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from backend.rag.embeddings import get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VECTOR_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "vector_db"
)

def create_vector_store(documents: List[Document]) -> Chroma:
    """
    Receives document chunks, generates embeddings, and stores them in a Chroma vector database.
    Persists the database locally to the `vector_db/` directory.

    Args:
        documents (List[Document]): The document chunks to store.
        
    Returns:
        Chroma: The initialized and populated Chroma vector store instance.
    """
    if not documents:
        logger.warning("No documents provided to create_vector_store. Returning empty store.")
        return Chroma(
            embedding_function=get_embedding_model(),
            persist_directory=VECTOR_DB_DIR
        )

    logger.info(f"Creating vector store for {len(documents)} document chunk(s)...")
    logger.info(f"Vector database will be persisted at: {VECTOR_DB_DIR}")

    try:
        embedding_model = get_embedding_model()
        
        # Initialize Chroma and add documents in one step
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=VECTOR_DB_DIR
        )
        
        logger.info(f"Successfully stored {len(documents)} document chunk(s) in the vector database.")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise

_vector_store_instance = None

def load_vector_store() -> Chroma:
    """
    Loads an existing persisted Chroma database from the local directory.
    Uses a global cache to ensure it only loads once per process.

    Returns:
        Chroma: The loaded Chroma vector store instance.
    """
    global _vector_store_instance
    if _vector_store_instance is not None:
        return _vector_store_instance
        
    logger.info(f"Loading existing vector store from: {VECTOR_DB_DIR}")
    
    if not os.path.exists(VECTOR_DB_DIR):
        logger.warning(f"Vector DB directory '{VECTOR_DB_DIR}' does not exist. It may be empty.")

    try:
        embedding_model = get_embedding_model()
        vector_store = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embedding_model
        )
        
        logger.info("Successfully loaded the vector store.")
        _vector_store_instance = vector_store
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise

def similarity_search(query: str, k: int = 4) -> List[Document]:
    """
    Embeds the provided query, searches the vector database, and returns 
    the most relevant document chunks.

    Args:
        query (str): The search query.
        k (int): The number of relevant document chunks to retrieve. Defaults to 4.

    Returns:
        List[Document]: The most relevant document chunks found in the database.
    """
    logger.info(f"Performing similarity search for query: '{query}' (top_k={k})")
    
    try:
        vector_store = load_vector_store()
        
        # Perform the actual search
        results = vector_store.similarity_search(query, k=k)
        
        logger.info(f"Successfully retrieved {len(results)} relevant document chunk(s).")
        return results
        
    except Exception as e:
        logger.error(f"Failed during similarity search: {e}")
        raise

if __name__ == "__main__":
    # Optional test block
    from backend.rag.document_loader import load_documents
    
    print("--- Testing Vector Store Creation ---")
    docs = load_documents()
    if docs:
        print(f"Loaded {len(docs)} docs, attempting to create vector store...")
        store = create_vector_store(docs)
        print("\n--- Testing Similarity Search ---")
        test_query = "What is the campus address?"
        search_results = similarity_search(test_query, k=2)
        if search_results:
            print(f"Top result for '{test_query}':\n{search_results[0].page_content[:200]}...")
        else:
            print(f"No results found for '{test_query}'")
