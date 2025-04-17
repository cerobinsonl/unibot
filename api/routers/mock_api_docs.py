from fastapi import APIRouter, HTTPException, Path, Query, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import json
import httpx
import os

router = APIRouter(prefix="/mock-api", tags=["mock-api"])

# Define environment variable for the agent system URL
AGENT_SYSTEM_URL = os.getenv("AGENT_SERVICE_URL", "http://agent-system:8000")

# Define Pydantic models for request/response validation and documentation

class ApiResponse(BaseModel):
    """Base response model for all API responses"""
    status: str
    message: str
    data: Any

# LMS Models
class CourseParameters(BaseModel):
    department: Optional[str] = Field(None, description="Filter courses by department")
    term: Optional[str] = Field("Fall2023", description="Academic term (e.g., 'Fall2023')")

class AssignmentParameters(BaseModel):
    course_id: str = Field(..., description="Course ID (e.g., 'CS101')")

class GradeParameters(BaseModel):
    course_id: str = Field(..., description="Course ID (e.g., 'CS101')")

class DiscussionParameters(BaseModel):
    course_id: str = Field(..., description="Course ID (e.g., 'CS101')")

# SIS Models
class EnrollmentParameters(BaseModel):
    department: Optional[str] = Field(None, description="Filter by department")
    year: Optional[str] = Field("2023", description="Academic year")

class TranscriptParameters(BaseModel):
    student_id: str = Field(..., description="Student ID")

class FinancialAidParameters(BaseModel):
    year: Optional[str] = Field("2023-2024", description="Financial aid year")

class DegreeProgressParameters(BaseModel):
    student_id: str = Field(..., description="Student ID")

# CRM Models
class ProspectiveStudentParameters(BaseModel):
    cycle: Optional[str] = Field("2023-2024", description="Application cycle")

class AlumniParameters(BaseModel):
    filters: Optional[Dict[str, Any]] = Field({}, description="Optional filters")

class DonationParameters(BaseModel):
    year: Optional[str] = Field("2023", description="Donation year")

class EventParameters(BaseModel):
    type: Optional[str] = Field("all", description="Event type")

# Helper function to call the agent system's mock API endpoint
async def call_agent_mock_api(system: str, endpoint: str, parameters: Dict[str, Any]):
    """Call the agent system's mock API endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AGENT_SYSTEM_URL}/mock-api/{system}/{endpoint}",
                json=parameters,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Endpoint not found: {system}/{endpoint}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling mock API: {str(e)}")

# === LMS API Routes ===

@router.post("/lms/courses", response_model=ApiResponse)
async def get_courses(parameters: CourseParameters):
    """Get courses from the Learning Management System"""
    return await call_agent_mock_api("lms", "courses", parameters.dict(exclude_none=True))

@router.post("/lms/assignments", response_model=ApiResponse)
async def get_assignments(parameters: AssignmentParameters):
    """Get assignments for a specific course from the LMS"""
    return await call_agent_mock_api("lms", "assignments", parameters.dict(exclude_none=True))

@router.post("/lms/grades", response_model=ApiResponse)
async def get_grades(parameters: GradeParameters):
    """Get grade distribution for a specific course from the LMS"""
    return await call_agent_mock_api("lms", "grades", parameters.dict(exclude_none=True))

@router.post("/lms/discussions", response_model=ApiResponse)
async def get_discussions(parameters: DiscussionParameters):
    """Get discussion activity for a specific course from the LMS"""
    return await call_agent_mock_api("lms", "discussions", parameters.dict(exclude_none=True))

# === SIS API Routes ===

@router.post("/sis/enrollment", response_model=ApiResponse)
async def get_enrollment(parameters: EnrollmentParameters):
    """Get enrollment statistics from the Student Information System"""
    return await call_agent_mock_api("sis", "enrollment", parameters.dict(exclude_none=True))

@router.post("/sis/transcript", response_model=ApiResponse)
async def get_transcript(parameters: TranscriptParameters):
    """Get a student's transcript from the SIS"""
    return await call_agent_mock_api("sis", "transcript", parameters.dict(exclude_none=True))

@router.post("/sis/financial-aid", response_model=ApiResponse)
async def get_financial_aid(parameters: FinancialAidParameters):
    """Get financial aid information from the SIS"""
    return await call_agent_mock_api("sis", "financial-aid", parameters.dict(exclude_none=True))

@router.post("/sis/degree-progress", response_model=ApiResponse)
async def get_degree_progress(parameters: DegreeProgressParameters):
    """Get degree progress information for a student from the SIS"""
    return await call_agent_mock_api("sis", "degree-progress", parameters.dict(exclude_none=True))

# === CRM API Routes ===

@router.post("/crm/prospective-students", response_model=ApiResponse)
async def get_prospective_students(parameters: ProspectiveStudentParameters):
    """Get prospective student data from the CRM"""
    return await call_agent_mock_api("crm", "prospective-students", parameters.dict(exclude_none=True))

@router.post("/crm/alumni", response_model=ApiResponse)
async def get_alumni(parameters: AlumniParameters):
    """Get alumni data from the CRM"""
    return await call_agent_mock_api("crm", "alumni", parameters.dict(exclude_none=True))

@router.post("/crm/donations", response_model=ApiResponse)
async def get_donations(parameters: DonationParameters):
    """Get donation data from the CRM"""
    return await call_agent_mock_api("crm", "donations", parameters.dict(exclude_none=True))

@router.post("/crm/events", response_model=ApiResponse)
async def get_events(parameters: EventParameters):
    """Get event data from the CRM"""
    return await call_agent_mock_api("crm", "events", parameters.dict(exclude_none=True))

# Generic endpoint for testing
@router.post("/{system}/{endpoint}")
async def call_mock_api(
    system: str = Path(..., description="System name (lms, sis, crm)"),
    endpoint: str = Path(..., description="API endpoint"),
    parameters: Dict[str, Any] = {}
):
    """Generic endpoint for calling any mock API connector directly"""
    return await call_agent_mock_api(system, endpoint, parameters)