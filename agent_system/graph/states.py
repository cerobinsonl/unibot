from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union

class Message(BaseModel):
    """A message in the conversation"""
    role: str
    content: str

class ConversationHistory(BaseModel):
    """Conversation history between user and assistant"""
    messages: List[Message] = []

class VisualizationData(BaseModel):
    """Data for a visualization"""
    image_data: str = Field(..., description="Base64 encoded image data")
    image_type: str = Field("image/png", description="MIME type of the image")
    title: Optional[str] = None
    description: Optional[str] = None

class IntermediateStep(BaseModel):
    """An intermediate step in the processing pipeline"""
    agent: str
    action: str
    input: Union[str, Dict[str, Any]]
    output: Union[str, Dict[str, Any], None] = None
    timestamp: str
    status: str = "completed"  # "started", "completed", "failed"

class SQLQueryResult(BaseModel):
    """Result of a SQL query"""
    query: str
    results: List[Dict[str, Any]]
    column_names: List[str]
    row_count: int
    
class AnalysisResult(BaseModel):
    """Result of data analysis"""
    summary: str
    details: Dict[str, Any]
    charts: Optional[List[str]] = None

class ApiCallResult(BaseModel):
    """Result of an API call to external system"""
    system: str
    endpoint: str
    response: Dict[str, Any]
    status: str

class EmailResult(BaseModel):
    """Result of an email sending operation"""
    recipients: List[str]
    subject: str
    status: str
    message_id: Optional[str] = None

class DatabaseOperationResult(BaseModel):
    """Result of a database operation"""
    operation: str  # "insert", "update", "delete"
    table: str
    affected_rows: int
    status: str

class DirectorState(BaseModel):
    """State for the Director Agent"""
    intent: str
    coordinator: str
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    final_response: Optional[str] = None

class CoordinatorState(BaseModel):
    """Base state for Coordinator Agents"""
    task: str
    specialists_needed: List[str]
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    result: Optional[Dict[str, Any]] = None

class SpecialistState(BaseModel):
    """Base state for Specialist Agents"""
    task: str
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    result: Optional[Dict[str, Any]] = None

# Specific Coordinator States
class DataAnalysisState(CoordinatorState):
    """State for Data Analysis Coordinator"""
    query_task: Optional[str] = None
    analysis_task: Optional[str] = None
    visualization_task: Optional[str] = None
    sql_result: Optional[SQLQueryResult] = None
    analysis_result: Optional[AnalysisResult] = None
    visualization_result: Optional[VisualizationData] = None

class CommunicationState(CoordinatorState):
    """State for Communication Coordinator"""
    message_type: str  # "email", "notification", etc.
    recipients: List[str]
    subject: Optional[str] = None
    content: str
    email_result: Optional[EmailResult] = None

class DataManagementState(CoordinatorState):
    """State for Data Management Coordinator"""
    operation_type: str  # "insert", "update", "delete", etc.
    table: str
    data: Dict[str, Any]
    validation_result: Optional[Dict[str, Any]] = None
    operation_result: Optional[DatabaseOperationResult] = None

class IntegrationState(CoordinatorState):
    """State for Integration Coordinator"""
    system: str  # "LMS", "SIS", "CRM", etc.
    endpoint: str
    parameters: Dict[str, Any]
    api_result: Optional[ApiCallResult] = None