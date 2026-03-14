from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import sys
import os

# Import the RAG pipeline function
from backend.rag.qa_chain import ask_question

# Configure unified logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI application instance
app = FastAPI(
    title="Smart Campus AI Assistant API",
    description="API server exposing the RAG-based AI Assistant for campus information queries.",
    version="1.0.0"
)

# Define Pydantic models for request structuring and validation
class ChatRequest(BaseModel):
    """
    Schema for an incoming user query.
    """
    query: str

class ChatResponse(BaseModel):
    """
    Schema for the generated API response.
    """
    query: str
    answer: str

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running correctly.
    
    Returns:
        dict: A JSON object with a status and descriptive message.
    """
    logger.info("Health check endpoint pinged.")
    return {
        "status": "ok",
        "message": "Smart Campus AI API is running"
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint to query the RAG pipeline.
    
    Expects a JSON payload containing the user's `query`. 
    Passes the query to the underlying language model and vector database, 
    and returns the generated answer.
    
    Args:
        request (ChatRequest): The incoming request payload.
        
    Returns:
        ChatResponse: The outgoing response payload with the query and its corresponding answer.
    """
    user_query = request.query.strip()
    
    if not user_query:
        logger.warning("Received empty query string.")
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    logger.info(f"Incoming Chat Request: '{user_query}'")
    
    try:
        # Utilize the implemented RAG chain asking functionality
        answer = ask_question(user_query)
        
        logger.info(f"Successfully generated answer for query: '{user_query}'")
        
        return ChatResponse(
            query=user_query,
            answer=answer
        )
        
    except Exception as e:
        logger.error(f"System Error while processing query '{user_query}': {e}", exc_info=True)
        # Raise HTTP 500 for internal pipeline failures
        raise HTTPException(
            status_code=500, 
            detail="The AI pipeline encountered an internal system error while processing the request."
        )

# For testing functionality when running script directly
if __name__ == "__main__":
    import uvicorn
    # Make sure we're binding to localhost with the standard fast api port
    logger.info("Starting up FastAPI application locally on port 8000...")
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
