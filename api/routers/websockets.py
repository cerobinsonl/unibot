from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
import logging
import json
import asyncio
import httpx
import os
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ws", tags=["websockets"])

# Agent system service URL
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://agent-system:8000")

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_message(self, session_id: str, message: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)
    
    async def send_json(self, session_id: str, data: Dict[str, Any]):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)

# Create connection manager instance
manager = ConnectionManager()

@router.websocket("/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat interactions
    """
    await manager.connect(session_id, websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                request_data = json.loads(data)
                
                # Add session ID if not present
                if "session_id" not in request_data:
                    request_data["session_id"] = session_id
                
                # Create a background task to process the request
                asyncio.create_task(
                    process_websocket_message(session_id, request_data)
                )
                
                # Send acknowledgment
                await manager.send_json(session_id, {
                    "type": "ack",
                    "message": "Processing your request..."
                })
                
            except json.JSONDecodeError:
                await manager.send_json(session_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(session_id)

async def process_websocket_message(session_id: str, request_data: Dict[str, Any]):
    """
    Process a message received via WebSocket and send updates
    """
    try:
        # Send processing status
        await manager.send_json(session_id, {
            "type": "status",
            "message": "Request received, processing..."
        })
        
        # Call agent system
        async with httpx.AsyncClient() as client:
            # Send the request to the agent system
            response = await client.post(
                f"{AGENT_SERVICE_URL}/ws/process",
                json=request_data,
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            
            # Send the result back to the client
            await manager.send_json(session_id, {
                "type": "result",
                **result
            })
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from agent system: {e}")
        await manager.send_json(session_id, {
            "type": "error",
            "message": f"Agent system error: {e.response.status_code}"
        })
    except httpx.RequestError as e:
        logger.error(f"Request error to agent system: {e}")
        await manager.send_json(session_id, {
            "type": "error",
            "message": "Agent system unavailable"
        })
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        await manager.send_json(session_id, {
            "type": "error",
            "message": "An unexpected error occurred"
        })

@router.websocket("/stream/{session_id}")
async def streaming_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming results for long-running operations
    """
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                request_data = json.loads(data)
                
                # Special handling for streaming data requests
                if request_data.get("type") == "stream_request":
                    asyncio.create_task(
                        handle_streaming_request(session_id, request_data)
                    )
                else:
                    await manager.send_json(session_id, {
                        "type": "error",
                        "message": "Invalid request type for streaming endpoint"
                    })
                
            except json.JSONDecodeError:
                await manager.send_json(session_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"Streaming WebSocket error: {e}")
        manager.disconnect(session_id)

async def handle_streaming_request(session_id: str, request_data: Dict[str, Any]):
    """
    Handle streaming data requests with progressive updates
    """
    try:
        # Connect to agent system with streaming support
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{AGENT_SERVICE_URL}/stream/process",
                json=request_data,
                timeout=300.0  # Longer timeout for streaming
            ) as response:
                response.raise_for_status()
                
                # Process the streaming response
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                        
                    try:
                        chunk = json.loads(line)
                        await manager.send_json(session_id, chunk)
                        
                        # If this is the final chunk, break
                        if chunk.get("status") == "complete":
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in streaming response: {line}")
                
        # Send completion message
        await manager.send_json(session_id, {
            "type": "status",
            "status": "complete",
            "message": "Processing complete"
        })
                
    except Exception as e:
        logger.error(f"Error in streaming request: {e}")
        await manager.send_json(session_id, {
            "type": "error",
            "message": str(e)
        })