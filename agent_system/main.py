from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import logging
import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Import graph workflow
from graph.workflow import create_workflow, AgentState
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not os.getenv("AGENT_DEBUG", "false").lower() == "true" else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Active sessions store
active_sessions: Dict[str, AgentState] = {}

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources
    logger.info("Starting agent system")
    yield
    # Shutdown: Cleanup resources
    logger.info("Shutting down agent system")

# Create FastAPI app
app = FastAPI(
    title="University Agent System",
    description="LangGraph-based multi-agent system for university administration",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/process")
async def process_request(request_data: Dict[str, Any]):
    """
    Process a request through the agent system
    """
    try:
        session_id = request_data.get("session_id")
        user_message = request_data.get("message")
        
        if not session_id or not user_message:
            raise HTTPException(status_code=400, detail="Missing session_id or message")
        
        # Get or create workflow for this session
        if session_id not in active_sessions:
            logger.info(f"Creating new workflow for session {session_id}")
            workflow = create_workflow()
            active_sessions[session_id] = AgentState(
                session_id=session_id,
                workflow=workflow,
                history=[]
            )
        
        state = active_sessions[session_id]
        
        # Process the message through the workflow
        result = await state.workflow.ainvoke({
            "user_input": user_message,
            "session_id": session_id,
            "history": state.history
        })
        
        # Update history
        state.history.append({"role": "user", "content": user_message})
        state.history.append({"role": "assistant", "content": result.get("response", "")})
        
        # Truncate history if it gets too long
        if len(state.history) > 20:  # Keep last 10 exchanges
            state.history = state.history[-20:]
        
        # Prepare response
        response = {
            "message": result.get("response", ""),
            "session_id": session_id
        }
        
        # Add visualization if present
        if "visualization" in result:
            response["image_data"] = result["visualization"]["image_data"]
            response["image_type"] = result["visualization"]["image_type"]
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ws/process")
async def websocket_process(request_data: Dict[str, Any]):
    """
    Process a request from WebSocket
    """
    try:
        # Similar to the process_request but formatted for WebSocket
        session_id = request_data.get("session_id")
        user_message = request_data.get("message")
        
        if not session_id or not user_message:
            return {"type": "error", "message": "Missing session_id or message"}
        
        # Get or create workflow for this session
        if session_id not in active_sessions:
            logger.info(f"Creating new workflow for session {session_id}")
            workflow = create_workflow()
            active_sessions[session_id] = AgentState(
                session_id=session_id,
                workflow=workflow,
                history=[]
            )
        
        state = active_sessions[session_id]
        
        # Process the message through the workflow
        result = await state.workflow.ainvoke({
            "user_input": user_message,
            "session_id": session_id,
            "history": state.history
        })
        
        # Update history
        state.history.append({"role": "user", "content": user_message})
        state.history.append({"role": "assistant", "content": result.get("response", "")})
        
        # Truncate history if it gets too long
        if len(state.history) > 20:
            state.history = state.history[-20:]
        
        # Prepare WebSocket response
        response = {
            "message": result.get("response", ""),
            "session_id": session_id
        }
        
        # Add visualization if present
        if "visualization" in result:
            response["image_data"] = result["visualization"]["image_data"]
            response["image_type"] = result["visualization"]["image_type"]
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing WebSocket request: {e}", exc_info=True)
        return {"type": "error", "message": str(e)}

@app.post("/stream/process")
async def stream_process(request_data: Dict[str, Any]):
    """
    Process a request with streaming response for long-running operations
    """
    async def generate_stream():
        try:
            session_id = request_data.get("session_id")
            user_message = request_data.get("message")
            
            if not session_id or not user_message:
                yield json.dumps({"type": "error", "message": "Missing session_id or message"}) + "\n"
                return
            
            # Get or create workflow
            if session_id not in active_sessions:
                logger.info(f"Creating new workflow for session {session_id}")
                workflow = create_workflow(streaming=True)  # Enable streaming mode
                active_sessions[session_id] = AgentState(
                    session_id=session_id,
                    workflow=workflow,
                    history=[]
                )
            
            state = active_sessions[session_id]
            
            # Send initial status
            yield json.dumps({
                "type": "status",
                "status": "processing",
                "message": "Starting processing..."
            }) + "\n"
            
            # Process with streaming
            result_generator = state.workflow.astream({
                "user_input": user_message,
                "session_id": session_id,
                "history": state.history,
                "stream": True
            })
            
            async for chunk in result_generator:
                # Format and yield each chunk
                yield json.dumps({
                    "type": "chunk",
                    "data": chunk
                }) + "\n"
                
                # Brief pause to avoid overwhelming the connection
                await asyncio.sleep(0.05)
            
            # Update history after streaming completes
            state.history.append({"role": "user", "content": user_message})
            # We'll need to reconstruct the full response from chunks for history
            # This would need to be implemented based on your specific chunk format
            
            # Send completion status
            yield json.dumps({
                "type": "status",
                "status": "complete",
                "message": "Processing complete"
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson"
    )

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    End a session and clean up resources
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"status": "success", "message": "Session ended"}
    else:
        return {"status": "error", "message": "Session not found"}

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("AGENT_DEBUG", "false").lower() == "true"
    )