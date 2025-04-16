from fastapi import APIRouter, HTTPException
import httpx
import logging
import os
import uuid
from typing import Dict, Any, Optional

# Import models
from models.requests import ChatRequest
from models.responses import ChatResponse, ChatResponseWithImage

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/chat", tags=["chat"])

# Agent system service URL
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://agent-system:8000")

# Helper function to call agent system
async def call_agent_system(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send request to the agent system and get response
    """
    try:
        logger.info(f"Calling agent system with request: {request_data}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/process",
                json=request_data,
                timeout=60.0  # Longer timeout for complex processing
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error calling agent system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main chat endpoint
@router.post("/message")
async def process_chat_message(request: ChatRequest):
    """
    Process a chat message from the user and return a response
    """
    logger.info(f"Received chat request: {request.message[:50]}...")
    
    # Prepare data for agent system
    agent_request = {
        "message": request.message,
        "session_id": request.session_id or str(uuid.uuid4()),
        "context": request.context or {}
    }
    
    # Call agent system
    response_data = await call_agent_system(agent_request)
    
    # Log full response for debugging
    logger.info(f"Agent response: {response_data}")
    
    # Check if visualization is present
    if "visualization" in response_data and response_data["visualization"]:
        # Return response with image
        visualization = response_data["visualization"]
        
        return {
            "message": response_data.get("message", ""),
            "session_id": response_data.get("session_id", agent_request["session_id"]),
            "image_data": visualization.get("image_data", ""),
            "image_type": visualization.get("image_type", "image/png"),
            "has_visualization": True
        }
    
    # Return standard response
    return {
        "message": response_data.get("message", ""),
        "session_id": response_data.get("session_id", agent_request["session_id"]),
        "has_visualization": False
    }

# Session management
@router.post("/session")
async def create_session():
    """
    Create a new chat session
    """
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    End a chat session
    """
    return {"status": "session ended", "session_id": session_id}