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

class SQLAgent:
    """
    SQL Agent is responsible for translating natural language queries into SQL
    and executing them against the university database.
    """
    
    def __init__(self):
        """Initialize the SQL Agent"""
        # Create the LLM using the helper function
        self.llm = get_llm("sql_agent")
        
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
        
        # Create the SQL generation prompt
        self.sql_prompt = f"""
You are the SQL Query Agent for a university administrative system.
Your specialty is translating natural language requests into SQL queries
for a PostgreSQL database containing university data.

University Database Schema:
{self.schema_info}

Format your response as a JSON object with these keys:
- sql_query: The PostgreSQL query to execute
- explanation: Brief explanation of what the query does and why

Ensure your SQL query:
1. Is valid PostgreSQL syntax
2. Uses appropriate joins and conditions
3. Includes proper handling of NULL values
4. Uses clear column aliases for readability
5. Is optimized for performance when possible
6. Is injection-safe (no string concatenation)

Example:
{{
  "sql_query": "SELECT d.name, COUNT(s.student_id) AS student_count FROM departments d LEFT JOIN students s ON d.department_id = s.major_id GROUP BY d.name ORDER BY student_count DESC;",
  "explanation": "This query counts the number of students in each department by joining the departments and students tables, grouping by department name, and ordering by student count in descending order."
}}

Task: {task}
"""
    
    def __call__(self, task: str) -> Dict[str, Any]:
        """
        Generate and execute a SQL query based on the natural language task
        
        Args:
            task: Natural language description of the data to retrieve
            
        Returns:
            Dictionary containing query results
        """
        try:
            # Generate SQL query
            formatted_prompt = self.sql_prompt.format(task=task)
            sql_response = self.llm.invoke(formatted_prompt).content
            
            # Parse the response
            try:
                # Try to parse as JSON
                parsed = json.loads(sql_response)
                sql_query = parsed.get("sql_query", "")
                explanation = parsed.get("explanation", "")
            except json.JSONDecodeError:
                # Extract query using regex if not valid JSON
                match = re.search(r'```sql\s*(.*?)\s*```', sql_response, re.DOTALL)
                if match:
                    sql_query = match.group(1)
                else:
                    # Last resort, try to find anything that looks like a SQL query
                    match = re.search(r'SELECT\s+.*?;', sql_response, re.DOTALL | re.IGNORECASE)
                    sql_query = match.group(0) if match else ""
                
                explanation = "Query extracted from non-JSON response."
            
            # Execute the query
            results, column_names = self.db.execute_query(sql_query)
            
            # Return results
            return {
                "query": sql_query,
                "explanation": explanation,
                "results": results,
                "column_names": column_names,
                "row_count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error in SQL Agent: {e}", exc_info=True)
            
            # For the POC, we'll return mock data when the real database fails
            # In production, this would return a proper error
            if os.getenv("MOCK_DATA_ON_ERROR", "true").lower() == "true":
                return self._generate_mock_data(task)
            
            raise e
    
    def _generate_mock_data(self, task: str) -> Dict[str, Any]:
        """
        Generate mock data for demonstration purposes
        
        Args:
            task: The original task that failed
            
        Returns:
            Dictionary with mock data
        """
        logger.info(f"Generating mock data for task: {task}")
        
        # Create a prompt for generating mock data
        mock_prompt = f"""
You are a database simulation agent. When a real database query fails, you generate realistic mock data 
that could have been returned by the query.

Analyze the query task and generate:
1. A JSON array of objects representing rows of data
2. A list of column names
3. A plausible query that might have been used

Format your response as a JSON object:
{{
  "mock_query": "The SQL query that would have been executed",
  "mock_column_names": ["col1", "col2", ...],
  "mock_results": [{{"col1": "value1", "col2": 123, ...}}, ...]
}}

Make the data realistic for a university setting, with appropriate data types, value ranges, and relationships.
Include approximately 10-20 rows of data.

Task: {task}
"""
        
        # Generate mock data
        try:
            mock_response = self.llm.invoke(mock_prompt).content
            mock_data = json.loads(mock_response)
            
            return {
                "query": mock_data.get("mock_query", "SELECT * FROM mock_data"),
                "explanation": "Mock data generated for demonstration",
                "results": mock_data.get("mock_results", []),
                "column_names": mock_data.get("mock_column_names", []),
                "row_count": len(mock_data.get("mock_results", [])),
                "is_mock": True
            }
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            
            # If parsing fails, return minimal mock data
            return {
                "query": "SELECT * FROM mock_data",
                "explanation": "Mock data generated for demonstration",
                "results": [{"id": 1, "name": "Mock Result"}],
                "column_names": ["id", "name"],
                "row_count": 1,
                "is_mock": True
            }