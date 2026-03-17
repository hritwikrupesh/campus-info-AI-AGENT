from dotenv import load_dotenv
import os

load_dotenv()

import logging
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from backend.rag.vector_store import load_vector_store

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Cache QA chain (prevents reloading model every request)
qa_chain = None


def get_llm() -> ChatGroq:
    """
    Initialize the fast hosted Groq API (llama-3.1-8b-instant).
    """
    logger.info("Initializing Groq Language Model Pipeline: llama-3.1-8b-instant")

    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not found.")

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            api_key=groq_api_key
        )

        logger.info("Successfully initialized Groq Language Model Pipeline.")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize Language Model: {e}")
        raise


def create_qa_chain() -> RetrievalQA:
    """
    Build the RetrievalQA chain using the vector database and Groq LLM.
    """

    logger.info("Initializing RetrievalQA Chain...")

    try:
        vector_store = load_vector_store()

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 25}
        )

        llm = get_llm()

        template = """
You are a Smart Campus AI Assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Give structured answers using clear formatting.
- Include complete lists where needed without omitting any items.
- Provide complete and detailed information based on the context.
- Avoid unnecessary "I don't know" responses if partial information exists in the context.
- Do NOT copy the context text directly.
- Do NOT repeat sections.

Context:
{context}

Question:
{question}

Answer:
"""

        prompt = PromptTemplate.from_template(template)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        logger.info("Successfully created RetrievalQA Chain.")
        return chain

    except Exception as e:
        logger.error(f"Failed to create QA Chain: {e}")
        raise


def clean_llm_output(answer: str) -> str:
    """
    Cleans messy LLM output and removes prompt leakage.
    """

    if not answer:
        return "I couldn't generate an answer."

    # Extract first answer
    if "Answer:" in answer:
        answer = answer.split("Answer:")[1]

    # Remove repeated sections
    stop_words = ["Context:", "Question:", "You are"]

    for word in stop_words:
        if word in answer:
            answer = answer.split(word)[0]

    # Remove duplicate answer loops
    answer = answer.strip()

    # Hard length limit to avoid runaway generation
    answer = answer[:700]

    return answer


def ask_question(query: str) -> dict:
    """
    Run the query through the RAG pipeline and return answer with sources.
    """

    logger.info(f"Received User Query: '{query}'")

    try:
        global qa_chain

        # Initialize QA chain only once
        if qa_chain is None:
            logger.info("QA chain not initialized. Creating now...")
            qa_chain = create_qa_chain()

        # Semantic query expansion handling to fix vocabulary mismatch
        # e.g., 'departments' in user queries vs 'programs' in actual college data documents
        search_query = query
        lower_query = query.lower()
        if "department" in lower_query or "branch" in lower_query:
            search_query += " programs branches undergraduate B.Tech postgraduate M.Tech offered"
        if "course" in lower_query:
            search_query += " programs curriculum"
        if "placement" in lower_query:
            search_query += " training recruitment"
        if "principal" in lower_query:
            search_query += " director administration"

        result = qa_chain.invoke({"query": search_query})

        raw_answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        answer = clean_llm_output(raw_answer)
        
        # Extract unique URLs from metadata robustly
        sources = []
        for doc in source_docs:
            url = doc.metadata.get("url")
            if url and url not in sources:
                sources.append(url)
                
        # Limit the number of sources to top 2
        sources = sources[:2]

        logger.info(f"Retrieved {len(source_docs)} relevant document chunk(s).")
        logger.info(f"Generated Response: {answer}")

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to generate answer for query '{query}': {e}")
        return {
            "answer": "Sorry, I encountered an error while processing your request.",
            "sources": []
        }


if __name__ == "__main__":
    import sys

    test_query = "What is the name of the campus?"

    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    print("\n--- Testing QA Chain ---")
    print(f"Question: {test_query}\n")

    try:
        response = ask_question(test_query)
        print(f"\nAnswer: {response['answer']}")
        print(f"Sources: {response['sources']}\n")

    except Exception as e:
        print(f"\nError running QA test: {e}\n")