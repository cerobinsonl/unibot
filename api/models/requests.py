from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
import uuid

class ChatRequest(BaseModel):
    """
    Model for chat message request
    """
    message: str = Field(..., description="The user's message")
    session_id: Optional[str] = Field(None, description="Session identifier, if continuing a conversation")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")

class DataQueryRequest(BaseModel):
    """
    Model for data query requests
    """
    query: str = Field(..., description="Natural language query about the data")
    session_id: Optional[str] = Field(None, description="Session identifier")
    visualization_type: Optional[str] = Field(None, description="Preferred visualization type if applicable")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters to apply")

class EmailRequest(BaseModel):
    """
    Model for email sending requests
    """
    recipients: List[str] = Field(..., description="Email recipients")
    subject: str = Field(..., description="Email subject")
    content: str = Field(..., description="Email content")
    session_id: Optional[str] = Field(None, description="Session identifier")

class DataEntryRequest(BaseModel):
    """
    Model for data entry requests
    """
    data: Dict[str, Any] = Field(..., description="Data to be entered")
    table: str = Field(..., description="Target table name")
    session_id: Optional[str] = Field(None, description="Session identifier")

class SyntheticDataRequest(BaseModel):
    """
    Model for synthetic data generation requests
    """
    data_schema: Dict[str, Any] = Field(..., description="Schema for synthetic data")
    amount: int = Field(..., description="Number of records to generate")
    session_id: Optional[str] = Field(None, description="Session identifier")