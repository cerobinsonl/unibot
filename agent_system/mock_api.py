from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json

# Import mock API connector functions
from tools.api_connectors import (
    call_lms_api,
    call_sis_api, 
    call_crm_api
)

router = APIRouter(prefix="/mock-api", tags=["mock-api"])

@router.post("/{system}/{endpoint}")
async def mock_api_endpoint(
    system: str = Path(..., description="System name (lms, sis, crm)"),
    endpoint: str = Path(..., description="API endpoint"),
    parameters: Dict[str, Any] = {}
):
    """
    Endpoint for accessing mock API data directly
    
    Args:
        system: External system name (lms, sis, crm)
        endpoint: API endpoint to call
        parameters: Parameters for the API call
        
    Returns:
        Mock API response data
    """
    try:
        if system.lower() == "lms":
            result = call_lms_api(endpoint, parameters)
            return result
        elif system.lower() == "sis":
            result = call_sis_api(endpoint, parameters)
            return result
        elif system.lower() == "crm":
            result = call_crm_api(endpoint, parameters)
            return result
        else:
            raise HTTPException(status_code=400, detail=f"Unknown system: {system}")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating mock data: {str(e)}"
        )