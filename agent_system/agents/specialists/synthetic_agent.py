import logging
from typing import Dict, List, Any, Optional
import json
import random
from datetime import datetime, timedelta
import pandas as pd

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Configure logging
logger = logging.getLogger(__name__)

class SyntheticAgent:
    """
    Synthetic Data Generator is responsible for creating realistic but fictional
    data for testing and demonstrations.
    """
    
    def __init__(self):
        """Initialize the Synthetic Data Generator"""
        # Create the LLM using the helper function
        self.llm = get_llm("synthetic_agent")
        
        # Create the schema analysis prompt
        self.schema_prompt = """
You are the Synthetic Data Generator for a university administrative system.
Your role is to create realistic but fictional data for testing and demonstrations.

You need to analyze a database schema and generate specifications for synthetic data generation.

Your task is to:
1. Identify the key fields that need values
2. Determine appropriate data ranges and formats for each field
3. Define relationships between tables if relevant
4. Create rules to ensure the data will be realistic

Format your response as a JSON object with these keys:
- fields: Object mapping field names to specifications
- relationships: Information about related tables/fields
- constraints: Rules the data must follow

Example:
{
  "fields": {
    "first_name": {"type": "name", "gender": "any"},
    "last_name": {"type": "surname"},
    "email": {"type": "email", "domain": "university.edu"},
    "enrollment_date": {"type": "date", "min": "2020-01-01", "max": "2023-12-31"},
    "status": {"type": "choice", "options": ["active", "inactive", "graduated", "leave of absence"]}
  },
  "relationships": {
    "major_id": {"table": "departments", "field": "department_id"}
  },
  "constraints": [
    "enrollment_date cannot be in the future",
    "if status is 'graduated', graduation_date must be populated"
  ]
}

Table: {table}
Schema: {schema}
Record count: {record_count}

Please analyze this schema and provide specifications for generating synthetic data.
"""
        
        # Create the data generation prompt
        self.generation_prompt = """
You are the Synthetic Data Generator for a university administrative system.
Your role is to create realistic but fictional data for testing and demonstrations.

You need to generate synthetic data based on provided specifications.

Your task is to create realistic but fictional data entries that meet these criteria:
1. Follow all field specifications and constraints
2. Create varied and realistic-looking values
3. Maintain internal consistency across records
4. Avoid any potentially sensitive or offensive content

Format your response as a JSON object with these keys:
- records: Array of generated data records

Example:
{
  "records": [
    {
      "first_name": "Emma",
      "last_name": "Johnson",
      "email": "ejohnson@university.edu",
      "enrollment_date": "2021-09-01",
      "status": "active"
    },
    {
      "first_name": "Michael",
      "last_name": "Smith",
      "email": "msmith@university.edu",
      "enrollment_date": "2020-09-01",
      "status": "active"
    }
  ]
}

Table: {table}
Specifications: {specifications}
Record count: {record_count}

Please generate synthetic data according to these specifications.
"""
    
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synthetic data based on schema information
        
        Args:
            input_data: Dictionary containing schema info and generation parameters
            
        Returns:
            Dictionary with generated data
        """
        try:
            # Extract information from input
            table = input_data.get("table", "students")
            schema = input_data.get("schema", {})
            record_count = input_data.get("record_count", 10)
            
            # If schema is provided, use it directly
            if schema:
                specifications = schema
            else:
                # Otherwise, analyze table structure to create specifications
                schema_text = self._get_schema_for_table(table)
                
                formatted_prompt = self.schema_prompt.format(
                    table=table,
                    schema=schema_text,
                    record_count=record_count
                )
                
                schema_response = self.llm.invoke(formatted_prompt)
                
                # Extract specifications from response
                content = schema_response.content
                
                try:
                    specifications = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use a default specification
                    specifications = self._get_default_specification(table)
            
            # Generate the synthetic data
            formatted_prompt = self.generation_prompt.format(
                table=table,
                specifications=json.dumps(specifications),
                record_count=min(record_count, 20)  # Limit to 20 records per generation to avoid token limits
            )
            
            generation_response = self.llm.invoke(formatted_prompt)
            
            # Extract generated data
            content = generation_response.content
            
            try:
                generation_result = json.loads(content)
                records = generation_result.get("records", [])
            except json.JSONDecodeError:
                # If not valid JSON, generate basic random data
                records = self._generate_basic_data(table, record_count)
            
            # If we need more records than the LLM generated in one go
            if record_count > len(records):
                # Generate more data using pandas and the patterns from existing records
                additional_records = self._expand_data_with_patterns(records, record_count - len(records))
                records.extend(additional_records)
            
            # Return the generated data
            return {
                "status": "success",
                "message": f"Generated {len(records)} records for {table}",
                "table": table,
                "record_count": len(records),
                "data": records
            }
            
        except Exception as e:
            logger.error(f"Error in Synthetic Agent: {e}", exc_info=True)
            
            # Generate basic random data as fallback
            fallback_data = self._generate_basic_data(
                input_data.get("table", "students"), 
                input_data.get("record_count", 10)
            )
            
            return {
                "status": "partial_success",
                "message": f"Error during optimal generation, falling back to basic data: {str(e)}",
                "table": input_data.get("table", "students"),
                "record_count": len(fallback_data),
                "data": fallback_data
            }