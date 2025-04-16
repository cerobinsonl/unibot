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
from utils.tracer import tracer
# Import graph workflow
from graph.workflow import create_workflow, AgentState
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum visibility
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
        
        logger.info(f"Processing request for session {session_id}: {user_message[:50]}...")
        
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
        logger.debug(f"Invoking workflow with message: {user_message[:50]}...")
        initial_state = {
            "user_input": user_message,
            "session_id": session_id,
            "history": state.history,
            "intermediate_steps": [],
            "visualization": None
        }
        
        result = await state.workflow.ainvoke(initial_state)
        logger.debug(f"Workflow result keys: {list(result.keys())}")
        
        # Update history
        if state.history is None:
            state.history = []
        
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
        
        # Add visualization if present - with detailed logging
        if "visualization" in result:
            if result["visualization"] is not None:
                logger.info(f"Found visualization in result: {result['visualization'].get('chart_type', 'unknown type')}")
                logger.debug(f"Visualization data keys: {list(result['visualization'].keys())}")
                response["visualization"] = result["visualization"]
            else:
                logger.info("Visualization key present but value is None")
        else:
            logger.info("No visualization key in result")
            
            # Check intermediate steps for visualization
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if step.get("agent") == "visualization_agent":
                        logger.info("Found visualization step in intermediate_steps")
                        if "output" in step and step["output"] != "Visualization created":
                            logger.info("Visualization output found in intermediate steps")
                            
                            # Try to extract visualization data
                            if isinstance(step["output"], dict) and "image_data" in step["output"]:
                                logger.info("Found image_data in visualization step output")
                                response["visualization"] = step["output"]
        
        # For safety, ensure visualization is at the response level
        vis_found = False
        if "visualization" in response:
            vis_found = True
        
        # If we didn't find a visualization but mentioned one in the response, create a dummy
        if not vis_found and "visualization" in result.get("response", "").lower():
            logger.warning("Visualization mentioned in response but no visualization data found!")
        
        logger.info(f"Final response keys: {list(response.keys())}")

        # Near the end of the process_request function, before returning the response
        try:
            # Record final result in tracer
            tracer_file = tracer.complete_trace(result)
            logger.info(f"Agent trace saved to {tracer_file}")
        except Exception as e:
            logger.error(f"Error completing trace: {e}")

        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))