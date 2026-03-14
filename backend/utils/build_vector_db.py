import logging

from backend.rag.document_loader import load_documents
from backend.rag.vector_store import create_vector_store

logging.basicConfig(level=logging.INFO)

def main():
    print("\nLoading documents from data folder...\n")

    documents = load_documents()

    print(f"\nLoaded {len(documents)} document chunks\n")

    print("Creating vector database...\n")

    create_vector_store(documents)

    print("\nVector database created successfully!\n")


if __name__ == "__main__":
    main()