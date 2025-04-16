import logging
from typing import Dict, List, Any, Optional
import json
import re
import os

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
        
        # Dynamically fetch the database schema on initialization
        self.schema_info = self._get_database_schema()
        
        # Create SQL generation prompt for data operations
        self.sql_prompt = """
You are the Data Entry Agent for a university administrative system.
Your specialty is safely inserting and updating data in the university database.

You need to create a SQL statement for a database operation. Your task is to:

1. Generate appropriate SQL for the operation type (INSERT, UPDATE, DELETE)
2. Ensure the SQL follows PostgreSQL syntax
3. Include data validation checks where appropriate
4. Handle potential NULL values correctly
5. Use literal values directly in the SQL (not parameterized queries with $1, $2, etc.)

University Database Schema:
{schema_info}

IMPORTANT GUIDELINES:
1. This is the ACTUAL schema from the database - use ONLY these tables and columns.
2. Always use double quotes around table and column names: "TableName"."ColumnName".
3. Only operate on tables that exist in the schema provided.
4. If you cannot perform the operation with the available schema, explain what's missing.
5. Never invent or assume tables or columns that aren't in the schema.
6. The database is PostgreSQL.
7. Pay close attention to the actual table and column names in the schema.
8. For INSERT operations, identify the correct table and columns from the schema.
9. For UPDATE and DELETE operations, ensure the condition is specific enough.
10. If the requested table doesn't exist, LOOK FOR AN APPROPRIATE ALTERNATIVE (e.g., use "Person" for student data).
11. DO NOT use parameterized queries with $1, $2, etc. Instead, include the actual values directly:
    - For strings: INSERT INTO "Person" ("FirstName") VALUES ('John')
    - For numbers: INSERT INTO "Table" ("NumericColumn") VALUES (123)
    - For nulls: INSERT INTO "Table" ("Column") VALUES (NULL)

Operation type: {operation_type}
Table: {table}
Data: {data}
Condition: {condition}

Reply with a JSON object containing:
- "sql": The PostgreSQL statement to execute
- "explanation": Brief explanation of what the operation does
- "validation_warnings": Any potential data issues that should be checked
- "actual_table": The actual table name being used (which may differ from the requested table if corrections were made)

For example:
{{
  "sql": "INSERT INTO \\"Person\\" (\\"FirstName\\", \\"LastName\\", \\"EmailAddress\\") VALUES ('John', 'Doe', 'john.doe@example.com')",
  "explanation": "Adding a new person record with name and email information",
  "validation_warnings": ["Ensure email is unique"],
  "actual_table": "Person"
}}
"""
    
    def _get_database_schema(self) -> str:
        """
        Dynamically retrieve and format the database schema
        
        Returns:
            Formatted database schema as a string
        """
        schema_info = []
        try:
            # Get all tables in the database
            tables = self.db.get_tables()
            logger.info(f"Found {len(tables)} tables in the database: {', '.join(tables)}")
            
            # For each table, get its schema definition
            for table in tables:
                columns = self.db.get_table_schema(table)
                
                # Format as CREATE TABLE statement
                table_def = f'CREATE TABLE "{table}" (\n'
                
                column_defs = []
                for column in columns:
                    col_name = column.get("column_name", "")
                    data_type = column.get("data_type", "")
                    max_length = column.get("character_maximum_length")
                    is_nullable = column.get("is_nullable", "YES")
                    
                    # Format column type with length if applicable
                    if max_length and data_type == 'character varying':
                        data_type = f"VARCHAR({max_length})"
                    
                    # Format nullable constraint
                    null_constraint = "NULL" if is_nullable == "YES" else "NOT NULL"
                    
                    # Add to columns list
                    column_defs.append(f'    "{col_name}" {data_type} {null_constraint}')
                
                table_def += ",\n".join(column_defs)
                table_def += "\n);"
                
                schema_info.append(table_def)
                
            return "\n\n".join(schema_info)
        
        except Exception as e:
            logger.error(f"Error retrieving database schema: {e}")
            # If we can't get the schema dynamically, return a basic description
            return """
CREATE TABLE "Person" (
    "PersonId" SERIAL PRIMARY KEY,
    "FirstName" VARCHAR(100) NOT NULL,
    "LastName" VARCHAR(100) NOT NULL,
    "EmailAddress" VARCHAR(255),
    "DateOfBirth" DATE NULL,
    "Gender" VARCHAR(20) NULL,
    "PhoneNumber" VARCHAR(50) NULL,
    "CreatedOn" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

    def _clean_json_response(self, response: str) -> str:
        """
        Clean the JSON response from the LLM by removing markdown formatting
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code block markers if present
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        return response
    
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
            
            # Log key information
            logger.info(f"Attempting {operation_type} operation on table '{table}'")
            logger.info(f"Data keys: {list(data.keys())}")
            
            # Generate SQL for the operation
            formatted_prompt = self.sql_prompt.format(
                schema_info=self.schema_info,
                operation_type=operation_type,
                table=table,
                data=json.dumps(data),
                condition=condition
            )
            
            sql_response = self.llm.invoke(formatted_prompt).content
            logger.info(f"SQL generation response: {sql_response[:500]}")
            
            # Parse the response
            try:
                # Clean the response by removing markdown formatting
                cleaned_response = self._clean_json_response(sql_response)
                logger.info(f"Cleaned JSON response: {cleaned_response[:500]}")
                
                # Try to parse as JSON
                parsed = json.loads(cleaned_response)
                sql_statement = parsed.get("sql", "")
                explanation = parsed.get("explanation", "")
                validation_warnings = parsed.get("validation_warnings", [])
                actual_table = parsed.get("actual_table", table)
                
                # Log the extracted SQL statement
                logger.info(f"Extracted SQL statement: {sql_statement}")
                
                # Log the table correction if it happened
                if actual_table != table:
                    logger.info(f"Table corrected from '{table}' to '{actual_table}'")
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parse error: {json_err} - Response: {cleaned_response[:500]}")
                
                # Extract SQL using regex if not valid JSON
                sql_match = re.search(r'"sql"\s*:\s*"([^"]+)"', sql_response)
                if sql_match:
                    sql_statement = sql_match.group(1)
                    logger.info(f"Extracted SQL using regex: {sql_statement}")
                else:
                    # Last resort, try to find anything that looks like SQL
                    if operation_type == "insert":
                        match = re.search(r'INSERT INTO\s+.*?;', sql_response, re.DOTALL | re.IGNORECASE)
                    elif operation_type == "update":
                        match = re.search(r'UPDATE\s+.*?;', sql_response, re.DOTALL | re.IGNORECASE)
                    elif operation_type == "delete":
                        match = re.search(r'DELETE FROM\s+.*?;', sql_response, re.DOTALL | re.IGNORECASE)
                    
                    sql_statement = match.group(0) if match else ""
                    logger.info(f"Extracted SQL using pattern matching: {sql_statement}")
                
                explanation = "SQL extracted from non-JSON response."
                validation_warnings = []
                actual_table = table
            
            # Check if we have a valid SQL statement
            if not sql_statement:
                logger.error(f"No valid SQL statement generated. LLM response: {sql_response[:500]}")
                
                # Check if there's an explanation about table not existing
                table_explanation = None
                if "table" in sql_response.lower() and "not exist" in sql_response.lower():
                    # Try to extract explanation
                    expl_match = re.search(r'(The table.*?not exist.*?\.)', sql_response, re.DOTALL | re.IGNORECASE)
                    if expl_match:
                        table_explanation = expl_match.group(1)
                
                return {
                    "status": "error",
                    "message": f"Could not generate valid SQL statement. {table_explanation or 'The requested table or columns may not exist in the database.'}",
                    "operation_type": operation_type,
                    "table": table,
                    "affected_rows": 0,
                    "sql": None
                }
            
            # Execute the SQL
            try:
                # Only execute if it's a supported operation
                if operation_type == "insert" or operation_type == "update" or operation_type == "delete":
                    # First check if the statement is an authorized type
                    cleaned_query = re.sub(r'^\s*(--.*?\n)*\s*', '', sql_statement, flags=re.DOTALL).strip().upper()
                    if (operation_type == "insert" and cleaned_query.startswith("INSERT")) or \
                       (operation_type == "update" and cleaned_query.startswith("UPDATE")) or \
                       (operation_type == "delete" and cleaned_query.startswith("DELETE")):
                        
                        logger.info(f"Executing SQL: {sql_statement}")
                        
                        # Execute the statement
                        with self.db.engine.connect() as connection:
                            from sqlalchemy import text
                            result = connection.execute(text(sql_statement))
                            connection.commit()
                            affected_rows = result.rowcount
                            
                            logger.info(f"SQL executed successfully. Affected rows: {affected_rows}")
                    else:
                        raise ValueError(f"SQL statement type does not match requested operation: {operation_type}")
                else:
                    raise ValueError(f"Unsupported operation type: {operation_type}")
                
                # Return the results
                return {
                    "status": "success",
                    "message": f"Successfully {operation_type}ed {affected_rows} row(s) in table '{actual_table}'",
                    "operation_type": operation_type,
                    "table": actual_table,
                    "affected_rows": affected_rows,
                    "sql": sql_statement,
                    "explanation": explanation,
                    "validation_warnings": validation_warnings
                }
            except Exception as db_error:
                logger.error(f"Database error executing statement: {db_error}")
                return {
                    "status": "error",
                    "message": f"Database error: {str(db_error)}",
                    "operation_type": operation_type,
                    "table": actual_table,
                    "affected_rows": 0,
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