from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Import specialists
from agents.specialists.data_entry_agent import DataEntryAgent
from agents.specialists.synthetic_agent import SyntheticAgent  # Import the SyntheticAgent

# Configure logging
logger = logging.getLogger(__name__)

class DataManagementCoordinator:
    """
    Data Management Coordinator handles all database operations including
    data entry, updates, and validation of data integrity.
    """
    
    def __init__(self):
        """Initialize the Data Management Coordinator"""
        # Create the LLM using the helper function
        self.llm = get_llm("data_management_coordinator")
        
        # Initialize specialist agents
        self.data_entry_agent = DataEntryAgent()
        self.synthetic_agent = SyntheticAgent()  # Add the synthetic agent
        
        # Create the task planning prompt
        # NOTE: All JSON example curly braces are escaped by doubling them {{ }}
        self.planning_prompt = """
You are the Data Management Coordinator for a university administrative system.
Your responsibility is to oversee all database operations including data entry and updates.

You need to create a plan for handling this data management request. Determine what type of operation is needed:

1. INSERT - Add new records to the database
2. UPDATE - Modify existing records
3. DELETE - Remove records (usually soft delete by changing status)
4. GENERATE - Create synthetic data for testing or demonstration purposes

IMPORTANT GUIDELINES:
- Our system uses a table called "Person" to store all people-related information, including students, faculty, and staff.
- When adding a new student, use the "Person" table, NOT a "students" table.
- Key fields in the Person table include: "FirstName", "LastName", "EmailAddress", "PhoneNumber", "Gender", "DateOfBirth"
- Pay close attention to correct casing of table and column names.
- For student-specific operations, the "Person" table is used along with role-specific tables.
- For GENERATE operations, determine if this is a request for synthetic data generation.
- For GENERATE operations, you should set "use_temp_table" to true to use temporary tables and avoid modifying production data.

Format your response as a JSON object with these keys:
- operation_type: "insert", "update", "delete", or "generate"
- table: Target table name (use actual table names from the database)
- data: Data structure to insert or update (key-value pairs)
- condition: For updates and deletes, the condition to identify records
- validation_rules: Rules that the data must satisfy
- record_count: For generate operations, how many records to create

Example for an insert:
{{
  "operation_type": "insert",
  "table": "Person",
  "data": {{
    "FirstName": "Jane",
    "LastName": "Smith",
    "EmailAddress": "jsmith@example.edu",
    "Gender": "Female",
    "PhoneNumber": "555-123-4567"
  }},
  "validation_rules": [
    "EmailAddress must be unique"
  ]
}}

Example for synthetic data generation:
{{
  "operation_type": "generate",
  "table": "Person",
  "record_count": 50,
  "data": {{
    "specific_requirements": "varied GPA distributions",
    "use_temp_table": true
  }},
  "validation_rules": [
    "Records should have realistic data",
    "Foreign key relationships must be maintained"
  ]
}}

User request: {user_input}
"""
        
        # Create the results synthesis prompt
        self.synthesis_prompt = """
You are the Data Management Coordinator for a university administrative system.
Your responsibility is to oversee all database operations including data entry and updates.

You are synthesizing the results from data management operations to create a response for the user.

Review the data operation request and the results of the database operation, then create a clear response 
that confirms what was done and provides any relevant details.

Your response should:
1. Confirm what type of operation was performed
2. Indicate which table was affected
3. Mention how many records were affected
4. Note any important data that was modified (without exposing sensitive information)
5. Offer to help with anything else

Be professional and concise, as appropriate for university administrative staff.

User request: {user_input}

Operation details: 
Type: {operation_type}
Table: {table}
Affected records: {affected_records}

Operation result: {result}

Create a response summarizing the action taken.
"""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the data management request by coordinating specialist agents
        
        Args:
            state: Current state of the conversation
            
        Returns:
            Updated state with data operation results
        """
        try:
            # Extract information from state
            user_input = state.get("user_input", "")
            intermediate_steps = state.get("intermediate_steps", [])
            
            # Step 1: Create a plan for handling the request
            formatted_prompt = self.planning_prompt.format(user_input=user_input)
            planning_response = self.llm.invoke(formatted_prompt).content
            
            # Parse the planning response
            try:
                plan = json.loads(planning_response)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, extract what we can
                import re
                
                op_type_match = re.search(r'"operation_type"\s*:\s*"([^"]+)"', planning_response)
                op_type = op_type_match.group(1) if op_type_match else "insert"
                
                table_match = re.search(r'"table"\s*:\s*"([^"]+)"', planning_response)
                table = table_match.group(1) if table_match else "Person"
                
                # Try to extract data object
                data_match = re.search(r'"data"\s*:\s*(\{[^}]+\})', planning_response)
                if data_match:
                    try:
                        data = json.loads(data_match.group(1))
                    except:
                        data = {"extracted_from_request": "true"}
                else:
                    data = {"extracted_from_request": "true"}
                
                # Try to extract condition
                condition_match = re.search(r'"condition"\s*:\s*"([^"]+)"', planning_response)
                condition = condition_match.group(1) if condition_match else None
                
                # Try to extract record count for generate operations
                record_count_match = re.search(r'"record_count"\s*:\s*(\d+)', planning_response)
                record_count = int(record_count_match.group(1)) if record_count_match else 10
                
                plan = {
                    "operation_type": op_type,
                    "table": table,
                    "data": data,
                    "condition": condition,
                    "record_count": record_count
                }
            
            # Add planning step to intermediate steps
            intermediate_steps.append({
                "agent": "data_management",
                "action": "create_plan",
                "input": user_input,
                "output": plan,
                "timestamp": self._get_timestamp()
            })
            
            # Step 2: Execute the data operation based on type
            operation_result = None
            
            if plan["operation_type"] == "generate":
                # This is a synthetic data generation request
                logger.info(f"Handling synthetic data generation request for {plan.get('record_count', 10)} records")
                
                # Call the synthetic agent
                operation_result = self.synthetic_agent({
                    "user_input": user_input,
                    "table": plan["table"],
                    "record_count": plan.get("record_count", 10),
                    "specific_requirements": plan.get("data", {}).get("specific_requirements", ""),
                    "use_temp_table": plan.get("data", {}).get("use_temp_table", True)  # Default to using temp tables
                })
                
                # Add synthetic data generation step to intermediate steps
                intermediate_steps.append({
                    "agent": "synthetic_agent",
                    "action": "generate_data",
                    "input": {
                        "table": plan["table"],
                        "record_count": plan.get("record_count", 10)
                    },
                    "output": operation_result,
                    "timestamp": self._get_timestamp()
                })
                
                # Update affected records count for synthesis
                affected_records = operation_result.get("executed_count", 0)
                
            else:
                # Handle regular data operations (insert, update, delete)
                operation_result = self.data_entry_agent({
                    "operation_type": plan["operation_type"],
                    "table": plan["table"],
                    "data": plan["data"],
                    "condition": plan.get("condition")
                })
                
                # Add operation step to intermediate steps
                intermediate_steps.append({
                    "agent": "data_entry_agent",
                    "action": f"execute_{plan['operation_type']}",
                    "input": {
                        "table": plan["table"],
                        "data": "Data object"  # Don't log full data for privacy/brevity
                    },
                    "output": operation_result,
                    "timestamp": self._get_timestamp()
                })
                
                # Update affected records count for synthesis
                affected_records = operation_result.get("affected_rows", 0)
            
            # Step 3: Synthesize results
            synthesis_input = {
                "user_input": user_input,
                "operation_type": plan["operation_type"],
                "table": plan["table"],
                "affected_records": affected_records,
                "result": operation_result.get("message", "Operation completed.")
            }
            
            formatted_prompt = self.synthesis_prompt.format(**synthesis_input)
            response = self.llm.invoke(formatted_prompt).content
            
            # Update state
            state["response"] = response
            state["intermediate_steps"] = intermediate_steps
            state["current_agent"] = "data_management"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Data Management Coordinator: {e}", exc_info=True)
            error_response = f"I encountered an error while processing your data management request: {str(e)}. Please try rephrasing your request or contact support if the issue persists."
            
            # Update state with error
            state["response"] = error_response
            state["current_agent"] = "data_management"
            
            return state
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        return datetime.now().isoformat()