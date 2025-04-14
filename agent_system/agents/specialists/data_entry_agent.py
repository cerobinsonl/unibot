import logging
from typing import Dict, List, Any, Optional
import json
import re
import os
import random

# Import database tools
from tools.database import DatabaseConnection

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Configure logging
logger = logging.getLogger(__name__)

class DataEntryAgent:
    """
    Data Entry Agent is responsible for safely inserting and updating data
    in the university database with proper validation.
    """
    
    def __init__(self):
        """Initialize the Data Entry Agent"""
        # Create the LLM using the helper function
        self.llm = get_llm("data_entry_agent")
        
        # Initialize database connection
        self.db = DatabaseConnection(settings.DATABASE_URL)
        
        # For the POC, include database schema information in the prompt
        # In production, this would be dynamically fetched
        self.schema_info = """
PostgreSQL Database Schema for University Administration:

-- Students Table
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    date_of_birth DATE,
    gender VARCHAR(50),
    address TEXT,
    phone VARCHAR(20),
    enrollment_date DATE,
    major_id INTEGER REFERENCES departments(department_id),
    graduation_date DATE,
    status VARCHAR(20) CHECK (status IN ('active', 'inactive', 'graduated', 'leave of absence'))
);

-- Faculty Table
CREATE TABLE faculty (
    faculty_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    department_id INTEGER REFERENCES departments(department_id),
    position VARCHAR(100),
    hire_date DATE,
    phone VARCHAR(20),
    status VARCHAR(20) CHECK (status IN ('active', 'on leave', 'retired', 'terminated'))
);

-- Departments Table
CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    code VARCHAR(10) UNIQUE NOT NULL,
    chair_id INTEGER,
    building VARCHAR(100),
    budget DECIMAL(15, 2),
    established_date DATE
);

-- Courses Table
CREATE TABLE courses (
    course_id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    department_id INTEGER REFERENCES departments(department_id),
    credits INTEGER,
    level VARCHAR(20) CHECK (level IN ('undergraduate', 'graduate'))
);

-- Sections Table (Course offerings)
CREATE TABLE sections (
    section_id SERIAL PRIMARY KEY,
    course_id INTEGER REFERENCES courses(course_id),
    faculty_id INTEGER REFERENCES faculty(faculty_id),
    semester VARCHAR(20),
    year INTEGER,
    room VARCHAR(50),
    schedule VARCHAR(100),
    capacity INTEGER,
    status VARCHAR(20) CHECK (status IN ('open', 'closed', 'cancelled'))
);

-- Enrollments Table
CREATE TABLE enrollments (
    enrollment_id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(student_id),
    section_id INTEGER REFERENCES sections(section_id),
    enrollment_date DATE,
    grade VARCHAR(2),
    status VARCHAR(20) CHECK (status IN ('active', 'dropped', 'completed'))
);
"""
        
        # Create SQL generation prompt for data operations
        self.sql_prompt = f"""
You are the Data Entry Agent for a university administrative system.
Your specialty is safely inserting and updating data in the university database.

You need to create a SQL statement for a database operation. Your task is to:

1. Generate appropriate SQL for the operation type (INSERT, UPDATE, DELETE)
2. Ensure the SQL follows PostgreSQL syntax
3. Include data validation checks where appropriate
4. Handle potential NULL values correctly
5. Use parameterized queries for safety

University Database Schema:
{self.schema_info}

Format your response as a JSON object with these keys:
- sql: The PostgreSQL statement to execute
- explanation: Brief explanation of what the operation does and any validation
- validation_warnings: Any potential data issues that should be checked

Example for INSERT:
{{
  "sql": "INSERT INTO students (first_name, last_name, email, enrollment_date, status) VALUES (:first_name, :last_name, :email, :enrollment_date, :status)",
  "explanation": "This statement inserts a new student record with the provided information",
  "validation_warnings": ["Ensure email is unique", "Check that enrollment_date is not in the future"]
}}

Example for UPDATE:
{{
  "sql": "UPDATE students SET status = :status, graduation_date = :graduation_date WHERE student_id = :student_id",
  "explanation": "This statement updates a student's status and graduation date",
  "validation_warnings": ["Verify student_id exists", "Ensure graduation_date is after enrollment_date"]
}}

Operation type: {operation_type}
Table: {table}
Data: {data}
Condition: {condition}

Please generate the appropriate SQL statement.
"""
    
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a data entry operation
        
        Args:
            input_data: Dictionary containing operation details
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Extract information from input
            operation_type = input_data.get("operation_type", "insert").lower()
            table = input_data.get("table", "")
            data = input_data.get("data", {})
            condition = input_data.get("condition", "")
            
            # Validate input
            if not table:
                raise ValueError("Table name is required")
            
            if operation_type not in ["insert", "update", "delete"]:
                raise ValueError(f"Invalid operation type: {operation_type}")
            
            if operation_type in ["update", "delete"] and not condition:
                raise ValueError(f"{operation_type.capitalize()} operation requires a condition")
            
            if operation_type in ["insert", "update"] and not data:
                raise ValueError(f"{operation_type.capitalize()} operation requires data")
            
            # Generate SQL for the operation
            formatted_prompt = self.sql_prompt.format(
                operation_type=operation_type,
                table=table,
                data=json.dumps(data),
                condition=condition
            )
            
            sql_response = self.llm.invoke(formatted_prompt).content
            
            # Parse the response
            try:
                # Try to parse as JSON
                parsed = json.loads(sql_response)
                sql_statement = parsed.get("sql", "")
                explanation = parsed.get("explanation", "")
                validation_warnings = parsed.get("validation_warnings", [])
            except json.JSONDecodeError:
                # Extract SQL using regex if not valid JSON
                match = re.search(r'```sql\s*(.*?)\s*```', sql_response, re.DOTALL)
                if match:
                    sql_statement = match.group(1)
                else:
                    # Last resort, try to find anything that looks like SQL
                    if operation_type == "insert":
                        match = re.search(r'INSERT INTO\s+.*?;', sql_response, re.DOTALL | re.IGNORECASE)
                    elif operation_type == "update":
                        match = re.search(r'UPDATE\s+.*?;', sql_response, re.DOTALL | re.IGNORECASE)
                    elif operation_type == "delete":
                        match = re.search(r'DELETE FROM\s+.*?;', sql_response, re.DOTALL | re.IGNORECASE)
                    
                    sql_statement = match.group(0) if match else ""
                
                explanation = "SQL extracted from non-JSON response."
                validation_warnings = []
            
            # In a real implementation, this would execute the actual SQL
            # For the POC, we'll just simulate database operations
            if os.getenv("MOCK_DB_OPERATIONS", "true").lower() == "true" or not sql_statement:
                # Mock database operation
                affected_rows = random.randint(1, 5)
                status = "success"
                message = f"Successfully {operation_type}ed {affected_rows} row(s)"
            else:
                # Execute the SQL
                # This is commented out for safety in the POC
                # result = self.db.execute_query(sql_statement)
                # affected_rows = result.rowcount
                # status = "success"
                # message = f"Successfully {operation_type}ed {affected_rows} row(s)"
                raise NotImplementedError("Real database operations not enabled for safety")
            
            # Return the results
            return {
                "status": status,
                "message": message,
                "operation_type": operation_type,
                "table": table,
                "affected_rows": affected_rows,
                "sql": sql_statement,
                "explanation": explanation,
                "validation_warnings": validation_warnings
            }
            
        except Exception as e:
            logger.error(f"Error in Data Entry Agent: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error performing {input_data.get('operation_type', 'operation')}: {str(e)}",
                "operation_type": input_data.get("operation_type", "unknown"),
                "table": input_data.get("table", "unknown"),
                "affected_rows": 0
            }