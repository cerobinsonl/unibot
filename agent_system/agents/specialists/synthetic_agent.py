import logging
from typing import Dict, List, Any, Optional
import json
import random
from datetime import datetime, timedelta
import pandas as pd

# Import database tools
from tools.database import DatabaseConnection

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
        
        # Initialize database connection for schema retrieval
        try:
            self.db = DatabaseConnection(settings.DATABASE_URL)
            self.db_initialized = True
            # Dynamically fetch the database schema on initialization
            self.schema_info = self._get_database_schema()
            schema_size = len(self.schema_info)
            table_count = self.schema_info.count('CREATE TABLE')
            logger.info(f"Retrieved database schema with {table_count} tables, schema size: {schema_size} chars")
        except Exception as e:
            logger.error(f"Error initializing database connection: {e}", exc_info=True)
            self.db_initialized = False
            self.schema_info = "Error: Could not retrieve database schema"
        
        # Create the schema analysis prompt
        self.schema_prompt = """
You are the Synthetic Data Generator for a university administrative system.
Your role is to create realistic but fictional data for testing and demonstrations.

You need to analyze a database schema and generate specifications for synthetic data generation.

Database Schema:
{schema_info}

IMPORTANT GUIDELINES:
1. This is the ACTUAL schema from the database - use ONLY these tables and columns.
2. Never invent or assume tables or columns that aren't in the schema.
3. Always check if the PersonId is auto-generated and DO NOT specify it manually.
4. Pay close attention to the actual table and column names in the schema.

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
            return "Error retrieving database schema: " + str(e)
    
    def _get_table_schema(self, table_name: str) -> str:
        """
        Get schema information for a specific table
        
        Args:
            table_name: Name of the table to get schema for
            
        Returns:
            Schema information for the table
        """
        try:
            # Get all schema statements
            schema_statements = self.schema_info.split("\n\n")
            
            # Find the statement for this table
            for statement in schema_statements:
                if f'CREATE TABLE "{table_name}"' in statement:
                    return statement
            
            return f"Schema for table {table_name} not found"
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return f"Error getting table schema: {str(e)}"
    
    def _parse_specific_requirements(self, user_input: str) -> str:
        """
        Extract specific requirements from user input
        
        Args:
            user_input: User's request
            
        Returns:
            Specific requirements string
        """
        requirements = []
        
        # Check for GPA distribution requirements
        if "gpa" in user_input.lower() or "grade" in user_input.lower():
            if "varied" in user_input.lower() or "different" in user_input.lower():
                requirements.append("varied GPA distributions")
            elif "high" in user_input.lower():
                requirements.append("high GPA values (3.0-4.0)")
            elif "low" in user_input.lower():
                requirements.append("low GPA values (1.0-2.5)")
            else:
                requirements.append("realistic GPA distributions")
        
        # Other potential requirements to check for
        if "gender" in user_input.lower() and "balanced" in user_input.lower():
            requirements.append("balanced gender distribution")
        
        if "department" in user_input.lower() or "program" in user_input.lower():
            requirements.append("diverse program/department distribution")
        
        # If no specific requirements found, use a general description
        if not requirements:
            requirements.append("realistic demographic and academic distributions")
            
        return ", ".join(requirements)
    
    def _generate_temp_table_sql(self) -> List[str]:
        """
        Generate SQL statements to create temporary tables for all relevant tables
        
        Returns:
            List of SQL statements to create temporary tables
        """
        create_statements = []
        try:
            # Get all tables from the database
            tables = self.db.get_tables()
            
            # Filter to relevant tables for student records - using exact case as in database
            student_related_tables = []
            for t in tables:
                if t.lower() in ['person', 'operationpersonrole', 'psstudentacademicrecord', 
                              'psstudentprogram', 'psstudentenrollment', 'psstudentemergencycontact',
                              'psstudentemployment']:
                    student_related_tables.append(t)
            
            logger.info(f"Creating temp tables for: {', '.join(student_related_tables)}")
            
            # For each table, create a temporary table with the same structure
            for table in student_related_tables:
                # Get the schema for this table to extract column names and types
                columns = self.db.get_table_schema(table)
                
                # Extract column definitions for the CREATE TABLE statement
                column_defs = []
                for column in columns:
                    col_name = column.get("column_name", "")
                    data_type = column.get("data_type", "")
                    max_length = column.get("character_maximum_length")
                    
                    # Format column type with length if applicable
                    if max_length and data_type == 'character varying':
                        data_type = f"VARCHAR({max_length})"
                    
                    # Add to columns list - use the exact column names from the database
                    column_defs.append(f"{col_name} {data_type}")
                
                # Create a fresh temporary table definition
                columns_sql = ", ".join(column_defs)
                
                # Create the temporary table
                drop_statement = f"DROP TABLE IF EXISTS temp_{table.lower()};"
                create_statement = f"CREATE TABLE temp_{table.lower()} ({columns_sql});"
                
                # Add statements to the list
                create_statements.append(drop_statement)
                create_statements.append(create_statement)
                
                # If this is the Person table (or matching case in db), we need to set up the primary key and sequence
                if table.lower() == "person":
                    # Identify the primary key column name with correct case
                    pk_col = next((col.get("column_name") for col in columns 
                                  if col.get("column_name", "").lower() == "personid"), "personid")
                    
                    create_statements.append(f"ALTER TABLE temp_{table.lower()} ADD PRIMARY KEY ({pk_col});")
                    create_statements.append(f"CREATE SEQUENCE IF NOT EXISTS temp_{table.lower()}_id_seq;")
                    create_statements.append(f"ALTER TABLE temp_{table.lower()} ALTER COLUMN {pk_col} SET DEFAULT nextval('temp_{table.lower()}_id_seq');")
            
            logger.info(f"Generated {len(create_statements)} SQL statements to create temporary tables")
            return create_statements
        
        except Exception as e:
            logger.error(f"Error generating temp table SQL: {e}")
            return []
    
    def _execute_sql_statements(self, sql_statements: List[str]) -> int:
        """
        Execute a list of SQL statements
        
        Args:
            sql_statements: List of SQL statements to execute
            
        Returns:
            Number of successfully executed statements
        """
        success_count = 0
        
        if not sql_statements:
            logger.warning("No SQL statements to execute")
            return 0
            
        # Log statements for debugging
        for i, stmt in enumerate(sql_statements[:5]):  # Log just a few statements
            logger.info(f"SQL Statement {i+1}: {stmt}")
        
        # Execute statements individually - no need for a large transaction
        for statement in sql_statements:
            try:
                with self.db.engine.connect() as connection:
                    # Start a transaction for this statement
                    transaction = connection.begin()
                    
                    try:
                        from sqlalchemy import text
                        # Execute the statement
                        connection.execute(text(statement))
                        success_count += 1
                        
                        # Commit each statement individually
                        transaction.commit()
                        
                        if success_count % 10 == 0:  # Log every 10 successful statements
                            logger.info(f"Successfully executed {success_count} statements")
                            
                    except Exception as stmt_error:
                        # Rollback on error
                        transaction.rollback()
                        logger.warning(f"Error executing statement: {stmt_error}")
                        # Continue with next statement
            except Exception as conn_error:
                logger.error(f"Connection error: {conn_error}")
        
        logger.info(f"Total successful SQL statements: {success_count} out of {len(sql_statements)}")
        return success_count
    
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synthetic data based on schema information and user request
        
        Args:
            input_data: Dictionary containing schema info and generation parameters
            
        Returns:
            Dictionary with generated data
        """
        try:
            # Extract information from input
            user_input = input_data.get("user_input", "")
            table = input_data.get("table", "Person")
            record_count = input_data.get("record_count", 10)
            use_temp_table = input_data.get("use_temp_table", True)  # Default to using temp tables
            
            # Extract specific requirements from user input
            specific_requirements = self._parse_specific_requirements(user_input)
            
            if not self.db_initialized:
                raise ValueError("Database connection not initialized. Cannot generate synthetic data.")
            
            # Get schema for the requested table
            table_schema = self._get_table_schema(table)
            
            # Determine if we should use temporary tables
            temp_table_prefix = "temp_" if use_temp_table else ""
            
            # Generate SQL to create temporary tables if needed
            setup_sql = []
            if use_temp_table:
                setup_sql = self._generate_temp_table_sql()
                
                # Execute setup SQL first
                if setup_sql:
                    logger.info(f"Executing {len(setup_sql)} setup SQL statements")
                    setup_count = self._execute_sql_statements(setup_sql)
                    logger.info(f"Successfully executed {setup_count} setup statements")
            
            # First, use the LLM to generate data directly with specific SQL statements
            # Adjust based on what worked best in your database
            formatted_prompt = """
You are a database expert for a university administrative system. You need to generate {record_count} synthetic student records with {specific_requirements}.

Here is the database schema:
{schema_info}

I want you to generate valid SQL INSERT statements that will add synthetic student data to the database. Please follow these important guidelines:

1. NEVER include a specific PersonId value in INSERT statements - this is an auto-incrementing field
2. Include appropriate data for FirstName, LastName, EmailAddress, DateOfBirth, Gender, PhoneNumber
3. Use valid SQL for PostgreSQL
4. DO NOT use double quotes around table and column names in your SQL statements
5. Each table should be prefixed with '{temp_table_prefix}' (e.g. INSERT INTO {temp_table_prefix}person)
6. For each student, create a complete set of records across relevant tables
7. All table and column names should be lowercase in the SQL statements (PostgreSQL is case-sensitive)

IMPORTANT: Provide your response ONLY as an array of SQL statements, with ONE statement per line, in valid JSON format.

Example (format only):
[
  "INSERT INTO {temp_table_prefix}person (firstname, lastname) VALUES ('John', 'Doe');",
  "INSERT INTO {temp_table_prefix}psstudentacademicrecord (personid, gpa) VALUES (CURRVAL('{temp_table_prefix}person_id_seq'), 3.5);"
]
"""
            
            # Format the prompt with all our variables
            final_prompt = formatted_prompt.format(
                schema_info=self.schema_info,
                record_count=record_count,
                specific_requirements=specific_requirements,
                temp_table_prefix=temp_table_prefix
            )
            
            # Generate SQL directly
            logger.info(f"Generating synthetic data SQL for {record_count} records with requirements: {specific_requirements}")
            response = self.llm.invoke(final_prompt)
            
            # Extract SQL statements from the response
            content = response.content
            sql_statements = []
            
            try:
                # Try to parse as JSON array of strings
                sql_statements = json.loads(content)
                if not isinstance(sql_statements, list):
                    # If it's not a list, it might be a JSON object with a statements field
                    if isinstance(sql_statements, dict) and "statements" in sql_statements:
                        sql_statements = sql_statements["statements"]
                    else:
                        raise ValueError("Response is not a list of SQL statements")
            except json.JSONDecodeError:
                # Try to extract SQL statements using regex
                import re
                sql_matches = re.findall(r'INSERT INTO [^;]+;', content)
                if sql_matches:
                    sql_statements = sql_matches
                else:
                    # More aggressive extraction for any SQL-like content
                    sql_statements = [line.strip() for line in content.split('\n') 
                                    if line.strip().startswith('INSERT INTO') and line.strip().endswith(';')]
            
            # Execute the SQL statements if there are any
            success_count = 0
            executed_statements = []
            
            if sql_statements:
                # Filter out statements that try to set PersonId explicitly
                filtered_statements = [stmt for stmt in sql_statements 
                                     if not ("personid" in stmt.lower() and 
                                             any(str(i) in stmt for i in range(10)))]
                
                logger.info(f"Preparing to execute {len(filtered_statements)} SQL statements")
                success_count = self._execute_sql_statements(filtered_statements)
                executed_statements = filtered_statements[:5]  # Just store a few for logging
            else:
                logger.warning("No SQL statements were generated")
            
            # Return the results
            generated_data = []
            # If we executed statements successfully, try to fetch some samples of what was added
            if success_count > 0:
                try:
                    with self.db.engine.connect() as connection:
                        from sqlalchemy import text
                        # Get some sample data that was just inserted
                        person_table = f"{temp_table_prefix}person"
                        sample_query = text(f"SELECT * FROM {person_table} ORDER BY personid DESC LIMIT 10")
                        result = connection.execute(sample_query)
                        # Convert to dictionaries
                        for row in result:
                            try:
                                generated_data.append(dict(row._mapping))
                            except:
                                # Fallback to simpler dictionary creation if mapping fails
                                row_dict = {}
                                for i, column in enumerate(result.keys()):
                                    row_dict[column] = row[i]
                                generated_data.append(row_dict)
                except Exception as e:
                    logger.error(f"Error fetching generated data samples: {e}")
            
            return {
                "status": "success" if success_count > 0 else "error",
                "message": f"Generated and inserted {success_count} synthetic records into {temp_table_prefix}person table and related tables",
                "table": f"{temp_table_prefix}{table}",
                "record_count": success_count,
                "executed_count": success_count,
                "data": generated_data[:10],  # Only return a sample of the data
                "executed_statements": executed_statements  # Include a few of the executed statements for debugging
            }
            
        except Exception as e:
            logger.error(f"Error in Synthetic Agent: {e}", exc_info=True)
            
            return {
                "status": "error",
                "message": f"Error generating synthetic data: {str(e)}",
                "table": input_data.get("table", "unknown"),
                "record_count": 0,
                "executed_count": 0,
                "data": []
            }