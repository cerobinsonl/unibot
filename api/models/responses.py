from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List

class ChatResponse(BaseModel):
    """
    Basic chat response model
    """
    message: str = Field(..., description="The response message")
    session_id: str = Field(..., description="Session identifier")
    action_required: Optional[bool] = Field(False, description="Whether user action is required")
    action_type: Optional[str] = Field(None, description="Type of action required, if any")

class ChatResponseWithImage(ChatResponse):
    """
    Chat response that includes an image (typically a visualization)
    """
    image_data: str = Field(..., description="Base64 encoded image data")
    image_type: str = Field("image/png", description="Image MIME type")
    image_title: Optional[str] = Field(None, description="Title of the image")
    image_description: Optional[str] = Field(None, description="Description of what the image shows")

class DataQueryResponse(BaseModel):
    """
    Response for data query requests
    """
    data: List[Dict[str, Any]] = Field(..., description="Query result data")
    session_id: str = Field(..., description="Session identifier")
    summary: Optional[str] = Field(None, description="Summary of the results")
    visualization_url: Optional[str] = Field(None, description="URL to visualization if available")

class EmailResponse(BaseModel):
    """
    Response for email sending requests
    """
    success: bool = Field(..., description="Whether the email was sent successfully")
    message: str = Field(..., description="Status message")
    session_id: str = Field(..., description="Session identifier")

class DataEntryResponse(BaseModel):
    """
    Response for data entry requests
    """
    success: bool = Field(..., description="Whether the data was entered successfully")
    message: str = Field(..., description="Status message")
    affected_rows: Optional[int] = Field(None, description="Number of rows affected")
    session_id: str = Field(..., description="Session identifier")

class SyntheticDataResponse(BaseModel):
    """
    Response for synthetic data generation requests
    """
    data: List[Dict[str, Any]] = Field(..., description="Generated synthetic data")
    session_id: str = Field(..., description="Session identifier")
    record_count: int = Field(..., description="Number of records generated")