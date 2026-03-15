from dotenv import load_dotenv
load_dotenv()

import logging
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate

from backend.rag.vector_store import load_vector_store

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Cache QA chain (prevents reloading model every request)
qa_chain = None


def get_llm() -> HuggingFacePipeline:
    """
    Initialize the local TinyLlama model using HuggingFace pipeline.
    """

    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    logger.info(f"Initializing Local Language Model Pipeline: {repo_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.2,
            repetition_penalty=1.2,
            do_sample=True,
            return_full_text=False
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        logger.info("Successfully initialized Local Language Model Pipeline.")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize Language Model: {e}")
        raise


def create_qa_chain() -> RetrievalQA:
    """
    Build the RetrievalQA chain using the vector database and LLM.
    """

    logger.info("Initializing RetrievalQA Chain...")

    try:
        vector_store = load_vector_store()

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        llm = get_llm()

        template = """
You are a Smart Campus AI Assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do NOT copy the context text directly.
- Summarize the information clearly.
- Do NOT repeat sections.
- Keep the answer concise (3–5 sentences).
- If the answer is not in the context, say you don't know.

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


def ask_question(query: str) -> str:
    """
    Run the query through the RAG pipeline.
    """

    logger.info(f"Received User Query: '{query}'")

    try:
        global qa_chain

        # Initialize QA chain only once
        if qa_chain is None:
            logger.info("QA chain not initialized. Creating now...")
            qa_chain = create_qa_chain()

        result = qa_chain.invoke({"query": query})

        raw_answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        answer = clean_llm_output(raw_answer)

        logger.info(f"Retrieved {len(source_docs)} relevant document chunk(s).")
        logger.info(f"Generated Response: {answer}")

        return answer

    except Exception as e:
        logger.error(f"Failed to generate answer for query '{query}': {e}")
        return "Sorry, I encountered an error while processing your request."


if __name__ == "__main__":
    import sys

    test_query = "What is the name of the campus?"

    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    print("\n--- Testing QA Chain ---")
    print(f"Question: {test_query}\n")

    try:
        response = ask_question(test_query)
        print(f"\nAnswer: {response}\n")

    except Exception as e:
        print(f"\nError running QA test: {e}\n")