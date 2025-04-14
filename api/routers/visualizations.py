from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import Response, JSONResponse
import httpx
import logging
import os
import base64
from typing import Dict, Any, Optional
import io

# Import models
from models.requests import DataQueryRequest
from models.responses import DataQueryResponse, ChatResponseWithImage

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/visualizations", tags=["visualizations"])

# Agent system service URL
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://agent-system:8000")

@router.post("/generate")
async def generate_visualization(request: DataQueryRequest):
    """
    Generate a visualization based on a natural language query about data
    """
    try:
        # Prepare request to agent system
        agent_request = {
            "task_type": "visualization",
            "query": request.query,
            "session_id": request.session_id,
            "visualization_type": request.visualization_type,
            "filters": request.filters
        }
        
        # Send request to agent system
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/process",
                json=agent_request,
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
        
        # Check if visualization was created
        if "visualization" not in result:
            return JSONResponse(
                status_code=400,
                content={"message": "No visualization could be generated for this query"}
            )
        
        # Return the visualization data
        return ChatResponseWithImage(
            message=result.get("message", "Visualization generated successfully"),
            session_id=request.session_id or "default-session",
            image_data=result["visualization"]["image_data"],
            image_type=result["visualization"]["image_type"],
            image_title=result["visualization"].get("title"),
            image_description=result["visualization"].get("explanation")
        )
        
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from agent system: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating visualization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@router.post("/raw/{image_format}")
async def get_raw_visualization(request: DataQueryRequest, image_format: str = "png"):
    """
    Get a raw visualization image in the specified format
    """
    try:
        # Validate image format
        if image_format not in ["png", "jpg", "svg"]:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Generate visualization using the other endpoint
        viz_response = await generate_visualization(request)
        
        # Extract base64 image data
        base64_data = viz_response.image_data
        image_data = base64.b64decode(base64_data)
        
        # Determine content type
        content_type = f"image/{image_format}"
        
        # Return raw image
        return Response(content=image_data, media_type=content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating raw visualization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating raw visualization: {str(e)}")

@router.get("/sample")
async def get_sample_visualization():
    """
    Get a sample visualization for testing
    """
    try:
        # Request a sample visualization from the agent system
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{AGENT_SERVICE_URL}/sample_visualization",
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
        
        # Return the sample visualization
        return ChatResponseWithImage(
            message="Sample visualization generated successfully",
            session_id="sample",
            image_data=result["image_data"],
            image_type=result["image_type"],
            image_title="Sample Visualization",
            image_description="This is a sample visualization for testing purposes"
        )
        
    except Exception as e:
        logger.error(f"Error generating sample visualization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating sample visualization: {str(e)}")