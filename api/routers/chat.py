from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
import httpx
import logging
import os
import uuid
from typing import List, Optional, Dict, Any

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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/process",
                json=request_data,
                timeout=60.0  # Longer timeout for complex processing
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from agent system: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        logger.error(f"Request error to agent system: {e}")
        raise HTTPException(status_code=503, detail="Agent system unavailable")
    except Exception as e:
        logger.error(f"Unexpected error calling agent system: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Main chat endpoint
@router.post("/message", response_model=ChatResponse)
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
    response = await call_agent_system(agent_request)
    
    # Check if response contains an image
    if "image_data" in response:
        return ChatResponseWithImage(
            message=response.get("message", ""),
            session_id=response.get("session_id", agent_request["session_id"]),
            image_data=response["image_data"],
            image_type=response.get("image_type", "image/png")
        )
    
    # Return text-only response
    return ChatResponse(
        message=response.get("message", ""),
        session_id=response.get("session_id", agent_request["session_id"])
    )

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
    # In a production system, we would clean up session resources here
    return {"status": "session ended", "session_id": session_id}