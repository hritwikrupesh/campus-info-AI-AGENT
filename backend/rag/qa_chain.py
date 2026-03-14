from dotenv import load_dotenv
load_dotenv()

import os
import logging
from typing import Dict, Any

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate

# For getting the vector store retriever
from backend.rag.vector_store import load_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_llm() -> HuggingFacePipeline:
    """
    Initializes and returns the language model to be used in the QA Chain.
    This implementation uses a local HuggingFacePipeline to avoid brittle remote API API issues
    like the 'InferenceClient' missing post method error in LangChain.
    
    Returns:
        HuggingFacePipeline: The initialized local language model wrapper.
    """
    # Using a smaller, efficient model like Microsoft's Phi-2 or a tiny Llama variant 
    # since we are moving to a pipeline approach instead of an external API. 
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0" is fast and lightweight for CPU/basic GPU RAG setups.
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    logger.info(f"Initializing Local Language Model Pipeline: {repo_id}")
    
    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        
        # Create a text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1, # Low temperature for factual RAG answering
            repetition_penalty=1.1,
            do_sample=True
        )
        
        # Wrap the pipeline in LangChain's HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        
        logger.info("Successfully initialized Local Language Model Pipeline.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Language Model: {e}")
        raise

def create_qa_chain() -> RetrievalQA:
    """
    Loads the vector database, creates a retriever, initializes the language model,
    and returns a RetrievalQA chain.
    
    Returns:
        RetrievalQA: The initialized QA chain ready for queries.
    """
    logger.info("Initializing RetrievalQA Chain...")
    
    try:
        # Load the Chroma database and convert it into a retriever
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4} # Retrieve the top 4 chunks
        )
        
        # Initialize the LLM
        llm = get_llm()
        
        # Define a precise instruction prompt
        template = """You are a helpful and precise Campus Information AI assistant. 
Use the following pieces of retrieved context to answer the question at the end. 
If you don't know the answer, just say that you don't know based on the provided context, don't try to make up an answer.
Keep the answer concise, accurate, and relevant to the user's query.

Context:
{context}

Question: {question}

Helpful Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # Build the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # 'stuff' puts all retrieved chunks into the prompt context
            retriever=retriever,
            return_source_documents=True, # Allows us to see which chunks were retrieved
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        logger.info("Successfully created RetrievalQA Chain.")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Failed to create QA Chain: {e}")
        raise

def ask_question(query: str) -> str:
    """
    Passes a user query to the QA chain, retrieves relevant documents, 
    and generates an answer.
    
    Args:
        query (str): The user's question.
        
    Returns:
        str: The generated answer from the language model.
    """
    logger.info(f"Received User Query: '{query}'")
    
    try:
        # Create the chain (in a production environment, this would ideally be cached/initialized once)
        qa_chain = create_qa_chain()
        
        # Execute the query through the RetrievalQA chain
        result = qa_chain.invoke({"query": query})
        
        # Extract the answer and source documents
        answer = result.get('result', "No answer could be generated.")
        source_docs = result.get('source_documents', [])

        if "Helpful Answer:" in answer:
            answer = answer.split("Helpful Answer:")[-1].strip()
        
        logger.info(f"Retrieved {len(source_docs)} relevant document chunk(s) to answer the query.")
        logger.info(f"Generated Response: {answer}")
        
        return answer
        
    except Exception as e:
        logger.error(f"Failed to generate answer for query '{query}': {e}")
        return f"I encountered an error trying to process your request. Details: {e}"

if __name__ == "__main__":
    import sys
    # Example test query
    test_query = "What is the name of the campus?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
        
    print(f"\n--- Testing QA Chain ---")
    print(f"Question: {test_query}\n")
    try:
        response = ask_question(test_query)
        print(f"\nAnswer: {response}\n")
    except Exception as e:
        print(f"\nError running QA test: {e}\n")
